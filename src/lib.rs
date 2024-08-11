use itertools::Itertools;
use minimizers::par::packed::{IntoBpIterator, Packed};
use std::{cmp::Ordering, ops::Range};
use sux::{
    bit_field_vec,
    dict::EliasFano,
    rank_sel::SelectAdaptConst,
    traits::{BitFieldSlice, BitFieldSliceCore, BitFieldSliceMut, IndexedSeq},
};

pub trait PhfBuilder<T> {
    type Phf: Phf<T>;
    fn build(&self, keys: &[T]) -> Self::Phf;
}

pub trait Phf<T> {
    fn hash(&self, key: T) -> Option<usize>; // hash
    fn max(&self) -> usize; // upper bound on hashes
    fn size(&self) -> usize;
}

impl<T: ptr_hash::KeyT> PhfBuilder<T> for ptr_hash::PtrHashParams {
    type Phf = ptr_hash::PtrHash<T>;
    fn build(&self, keys: &[T]) -> Self::Phf {
        ptr_hash::PtrHash::new(keys, *self)
    }
}

impl<T: ptr_hash::KeyT> Phf<T> for ptr_hash::PtrHash<T> {
    fn hash(&self, key: T) -> Option<usize> {
        Some(self.index(&key))
    }
    fn max(&self) -> usize {
        self.max_index()
    }
    fn size(&self) -> usize {
        let bits = self.bits_per_element();
        ((bits.0 + bits.1) / 8. * self.n() as f32) as usize
    }
}

pub trait Minimizer {
    fn minimizer_one(&self, window: impl IntoBpIterator) -> (usize, u64); // (pos, value)
    fn minimizers(&self, text: impl IntoBpIterator) -> impl Iterator<Item = (usize, u64)>; // (pos, value)
    fn k(&self) -> usize;
    fn w(&self) -> usize;
    fn l(&self) -> usize {
        self.k() + self.w() - 1
    }
}

pub trait BpStorage {
    type BpSlice<'a>: IntoBpIterator
    where
        Self: 'a;
    fn concat(data: &[Self::BpSlice<'_>]) -> Self;
    fn get(&self) -> Self::BpSlice<'_>;
    fn slice(&self, range: Range<usize>) -> Self::BpSlice<'_>;
    fn size(&self) -> usize;
    #[cfg(test)]
    fn random(n: usize, alphabet: usize) -> Self;
}

impl BpStorage for Vec<u8> {
    type BpSlice<'a> = &'a [u8];

    fn concat(data: &[Self::BpSlice<'_>]) -> Self {
        data.concat()
    }
    fn get(&self) -> Self::BpSlice<'_> {
        &*self
    }
    fn slice(&self, range: Range<usize>) -> Self::BpSlice<'_> {
        &self[range]
    }
    fn size(&self) -> usize {
        size_of_val(self.as_slice())
    }
    #[cfg(test)]
    fn random(n: usize, alphabet: usize) -> Self {
        (0..n)
            .map(|_| ((rand::random::<u8>() as usize) % alphabet) as u8)
            .collect_vec()
    }
}

#[derive(Debug)]
pub struct PackedVec {
    pub seq: Vec<u8>,
    pub len: usize,
}

impl BpStorage for PackedVec {
    type BpSlice<'a> = Packed<'a>;

    fn concat(seqs: &[Self::BpSlice<'_>]) -> Self {
        let mut seq = vec![];
        seq.reserve(seqs.iter().map(|p| p.seq.len()).sum::<usize>());
        seq.extend(seqs.iter().flat_map(|p| p.seq.iter()));
        let len = seqs.iter().map(|x| x.len()).sum::<usize>();
        Self { seq, len }
    }

    fn get(&self) -> Self::BpSlice<'_> {
        Packed {
            seq: &self.seq,
            offset: 0,
            len: self.len,
        }
    }
    fn slice(&self, range: Range<usize>) -> Self::BpSlice<'_> {
        let mut p = Packed {
            seq: &self.seq,
            offset: range.start,
            len: range.len(),
        };
        p.normalize();
        p
    }
    fn size(&self) -> usize {
        size_of_val(self.seq.as_slice())
    }
    #[cfg(test)]
    fn random(n: usize, alphabet: usize) -> Self {
        assert!(alphabet == 4);
        let seq = (0..n.div_ceil(4))
            .map(|_| rand::random::<u8>())
            .collect_vec();
        PackedVec { seq, len: n }
    }
}

pub struct NtMinimizer {
    pub k: usize,
    pub w: usize,
}

impl Minimizer for NtMinimizer {
    fn minimizer_one(&self, window: impl IntoBpIterator) -> (usize, u64) {
        let pos = minimizers::par::minimizer::minimizer_window::<false>(window, self.k);
        let val = window.sub_slice(pos, self.k).to_word() as u64;
        (pos, val)
    }

    fn minimizers(&self, text: impl IntoBpIterator) -> impl Iterator<Item = (usize, u64)> {
        minimizers::par::minimizer::minimizer_scalar_it::<false>(text, self.k, self.w).map(
            move |pos| {
                let val = text.sub_slice(pos as usize, self.k).to_word() as u64;
                (pos as usize, val)
            },
        )
    }

    fn k(&self) -> usize {
        self.k
    }

    fn w(&self) -> usize {
        self.w
    }
}

// TODO: sparse index
// TODO: construction time/memory benchmarks
// TODO: query throughput benchmarks
pub struct SsHash<H: Phf<u64>, M: Minimizer, P: BpStorage> {
    minimizer: M,
    text: P,               // Todo bitpacked
    endpoints: Vec<usize>, // Todo EF
    num_uniq_minis: usize,
    phf: H,
    sizes: sux::dict::EliasFano<SelectAdaptConst<sux::bits::BitVec<Box<[usize]>>>>,
    offsets: sux::bits::BitFieldVec,
}

impl<H: Phf<u64>, M: Minimizer, P: BpStorage> SsHash<H, M, P> {
    // Convenience wrapper around build_from_slice that doesn't require type annotations.
    pub fn build(text: &[&P], minimizer: M, phf_builder: impl PhfBuilder<u64, Phf = H>) -> Self {
        let text = text.iter().map(|x| x.get()).collect_vec();
        Self::build_from_slice(&text, minimizer, phf_builder)
    }
    pub fn build_from_slice(
        text: &[P::BpSlice<'_>],
        minimizer: M,
        phf_builder: impl PhfBuilder<u64, Phf = H>,
    ) -> Self {
        let endpoints = text
            .iter()
            .scan(0, |acc, x| {
                *acc += x.len().next_multiple_of(4);
                Some(*acc)
            })
            .collect_vec();
        let text = P::concat(text);

        // Todo split at input seq boundaries
        let mut minis = minimizer.minimizers(text.get()).dedup().collect_vec();
        minis.sort_by_key(|x| x.1);
        // Todo: is passing an iterator to the phf builder sufficient?
        let uniq_mini_vals = minis.iter().map(|x| x.1).dedup().collect_vec();
        let num_uniq_minis = uniq_mini_vals.len();
        let phf = phf_builder.build(&uniq_mini_vals);
        let mut sizes = vec![0; phf.max() + 1];
        for (mini_val, chunk) in minis.iter().chunk_by(|x| x.1).into_iter() {
            if let Some(hash) = phf.hash(mini_val) {
                sizes[hash] = chunk.count();
            }
        }
        sizes = sizes
            .into_iter()
            .scan(0, |acc, x| {
                let out = *acc;
                *acc += x;
                Some(out)
            })
            .collect_vec();

        let mut sizes_ef = sux::dict::EliasFanoBuilder::new(sizes.len(), *sizes.last().unwrap());
        for &size in &sizes {
            sizes_ef.push(size);
        }

        let offset_bits = text.get().len().ilog2() as usize + 1;
        let mut offsets = bit_field_vec![offset_bits; minis.len(); 0];
        for (mini_val, chunk) in minis.iter().chunk_by(|x| x.1).into_iter() {
            if let Some(hash) = phf.hash(mini_val) {
                let start = sizes[hash] as usize;
                for (i, (pos, _)) in chunk.enumerate() {
                    offsets.set(start + i, *pos);
                }
            }
        }

        let sizes = sizes_ef.build();
        let sizes = unsafe { sizes.map_high_bits(SelectAdaptConst::<_, _>::new) };

        Self {
            minimizer,
            text,
            endpoints,
            num_uniq_minis,
            phf,
            sizes,
            offsets,
        }
    }

    // TODO: query_stream
    pub fn query_one(&self, window: P::BpSlice<'_>) -> Option<(usize, usize)> {
        let (minimizer_pos, minimizer) = self.minimizer.minimizer_one(window);
        let hash: usize = self.phf.hash(minimizer)?;
        let offsets_start = self.sizes.get(hash) as usize;
        let offsets_end = self.sizes.get(hash + 1) as usize;
        for idx in offsets_start..offsets_end {
            let offset = self.offsets.get(idx);
            if offset < minimizer_pos {
                continue;
            }
            let lmer_pos = offset as usize - minimizer_pos;
            if lmer_pos + self.minimizer.l() > self.text.get().len() {
                continue;
            }
            let lmer = &self.text.slice(lmer_pos..lmer_pos + self.minimizer.l());
            if lmer.to_word() == window.to_word() {
                return Some((
                    lmer_pos,
                    self.endpoints
                        .binary_search_by(|&x| {
                            if x < offset {
                                Ordering::Less
                            } else {
                                Ordering::Greater
                            }
                        })
                        .unwrap_err(),
                ));
            }
        }
        None
    }

    pub fn print_size(&self) {
        use size::Size;
        use std::mem::size_of_val;
        // All sizes in bytes.
        let text = self.text.size();
        let endpoints = size_of_val(self.endpoints.as_slice());
        let phf = self.phf.size();
        // TODO: Include size of rank-select structures.
        let sizes = EliasFano::<(), ()>::estimate_size(self.offsets.len(), self.sizes.len()) / 8;
        let offsets = self.offsets.len() * self.offsets.bit_width() / 8;
        let total = text + endpoints + phf + sizes + offsets;

        let num_bp = self.text.get().len() as f32;
        let num_minis = self.offsets.len() as f32;

        eprintln!(
            "part       {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "len", "max", "memory", "bits/kmer", "bits/mini", "bits/uniq mini"
        );
        eprintln!(
            "text:      {:>10} {:>10} {:>10} {:>10.1} {:>10.1} {:>10.1}",
            self.text.get().len(),
            3,
            format!("{}", Size::from_bytes(text)),
            8. * text as f32 / num_bp,
            8. * text as f32 / num_minis,
            8. * text as f32 / self.num_uniq_minis as f32
        );
        eprintln!(
            "endpoints: {:>10} {:>10} {:>10} {:>10.1} {:>10.1} {:>10.1}",
            self.endpoints.len(),
            self.endpoints.last().unwrap(),
            format!("{}", Size::from_bytes(endpoints)),
            8. * endpoints as f32 / num_bp,
            8. * endpoints as f32 / num_minis,
            8. * endpoints as f32 / self.num_uniq_minis as f32
        );
        eprintln!(
            "phf:       {:>10} {:>10} {:>10} {:>10.1} {:>10.1} {:>10.1}",
            self.num_uniq_minis,
            self.phf.max(),
            format!("{}", Size::from_bytes(phf)),
            8. * phf as f32 / num_bp,
            8. * phf as f32 / num_minis,
            8. * phf as f32 / self.num_uniq_minis as f32
        );
        eprintln!(
            "sizes:     {:>10} {:>10} {:>10} {:>10.1} {:>10.1} {:>10.1}",
            self.sizes.len(),
            self.sizes.get(self.phf.max()),
            format!("{}", Size::from_bytes(sizes)),
            8. * sizes as f32 / num_bp,
            8. * sizes as f32 / num_minis,
            8. * sizes as f32 / self.num_uniq_minis as f32
        );
        eprintln!(
            "offsets:   {:>10} {:>10} {:>10} {:>10.1} {:>10.1} {:>10.1}",
            self.offsets.len(),
            self.text.get().len(),
            format!("{}", Size::from_bytes(offsets)),
            8. * offsets as f32 / num_bp,
            8. * offsets as f32 / num_minis,
            8. * offsets as f32 / self.num_uniq_minis as f32
        );
        eprintln!(
            "total:     {:>10} {:>10} {:>10} {:>10.1} {:>10.1} {:>10.1}",
            "",
            "",
            format!("{}", Size::from_bytes(total)),
            8. * total as f32 / num_bp,
            8. * total as f32 / num_minis,
            8. * total as f32 / self.num_uniq_minis as f32
        );
    }
}

#[cfg(test)]
mod naive;

#[cfg(test)]
mod test {
    use super::*;
    use naive::*;

    fn test_pos<P: BpStorage>(
        minimizer: impl Minimizer,
        phf_builder: impl PhfBuilder<u64>,
        alphabet: usize,
    ) {
        let n = 100;
        let text = P::random(n, alphabet);
        let l = minimizer.k() + minimizer.w() - 1;
        let sshash = SsHash::build(&[&text], minimizer, phf_builder);

        // The alphabet is large enough that we assume no duplicate k-mers occur.
        for i in 0..text.get().len() - l {
            let window = text.slice(i..i + l);
            assert_eq!(sshash.query_one(window), Some((i, 0)));
        }
    }

    fn test_neg<P: BpStorage + std::fmt::Debug>(
        minimizer: impl Minimizer,
        phf_builder: impl PhfBuilder<u64>,
        alphabet: usize,
    ) {
        let n = 10000;
        let text = P::random(n, alphabet);
        let l = minimizer.l();
        let sshash = SsHash::build(&[&text], minimizer, phf_builder);

        // The alphabet is large enough that we assume no duplicate k-mers occur.
        for _ in 0..text.get().len() {
            let window = P::random(l, alphabet);
            assert_eq!(sshash.query_one(window.get()), None);
        }
    }

    #[test]
    fn naive_pos() {
        let minimizer = NaiveMinimizer { k: 6, w: 3 };
        let phf_builder = NaivePhfBuilder::new();
        test_pos::<Vec<_>>(minimizer, phf_builder, 256);
    }

    #[test]
    fn naive_neg() {
        let minimizer = NaiveMinimizer { k: 6, w: 3 };
        let phf_builder = NaivePhfBuilder::new();
        test_neg::<Vec<_>>(minimizer, phf_builder, 256);
    }

    #[test]
    fn ptrhash_pos() {
        let minimizer = NaiveMinimizer { k: 6, w: 3 };
        let phf_builder = ptr_hash::PtrHashParams {
            remap: false,
            ..Default::default()
        };
        test_pos::<Vec<_>>(minimizer, phf_builder, 256);
    }

    #[test]
    fn ptrhash_neg() {
        let minimizer = NaiveMinimizer { k: 6, w: 3 };
        let phf_builder = ptr_hash::PtrHashParams {
            remap: false,
            ..Default::default()
        };
        test_neg::<Vec<_>>(minimizer, phf_builder, 256);
    }

    #[test]
    fn ntmini_pos_packed() {
        let minimizer = NtMinimizer { k: 19, w: 11 };
        let phf_builder = NaivePhfBuilder::new();
        test_pos::<PackedVec>(minimizer, phf_builder, 4);
    }

    #[test]
    fn ntmini_neg_packed() {
        let minimizer = NtMinimizer { k: 19, w: 11 };
        let phf_builder = NaivePhfBuilder::new();
        test_neg::<PackedVec>(minimizer, phf_builder, 4);
    }

    #[test]
    fn ntmini_ptrhash_pos_packed() {
        let minimizer = NtMinimizer { k: 19, w: 11 };
        let phf_builder = ptr_hash::PtrHashParams {
            remap: false,
            ..Default::default()
        };
        test_pos::<PackedVec>(minimizer, phf_builder, 4);
    }

    #[test]
    fn ntmini_ptrhash_neg_packed() {
        let minimizer = NtMinimizer { k: 19, w: 11 };
        let phf_builder = ptr_hash::PtrHashParams {
            remap: false,
            ..Default::default()
        };
        test_neg::<PackedVec>(minimizer, phf_builder, 4);
    }

    #[ignore]
    #[test]
    fn print_size() {
        let text = PackedVec::random(10000000, 4);
        let k = 21;
        let w = 11;
        let minimizer = NtMinimizer { k, w };
        let phf_builder = ptr_hash::PtrHashParams {
            remap: false,
            ..Default::default()
        };

        let sshash = SsHash::build(&[&text], minimizer, phf_builder);
        sshash.print_size();
    }

    #[ignore]
    #[test]
    fn print_size_hg() {
        eprintln!("Reading..");
        let start = std::time::Instant::now();
        let mut packed_text = vec![];
        let Ok(mut reader) = needletail::parse_fastx_file("human-genome.fa") else {
            panic!("Did not find human-genome.fa. Add/symlink it to test runtime on it.");
        };
        while let Some(r) = reader.next() {
            let r = r.unwrap();
            eprintln!(
                "Read {:?} of len {:?}",
                std::str::from_utf8(r.id()),
                r.raw_seq().len()
            );
            pack(&r.raw_seq(), &mut packed_text);
            eprintln!("Packed len {:?}", packed_text.len());
        }
        eprintln!("Packing took {:?}", start.elapsed());

        let text = PackedVec {
            len: packed_text.len() * 4,
            seq: packed_text,
        };
        let k = 20;
        let w = 12;
        let minimizer = NtMinimizer { k, w };
        let phf_builder = ptr_hash::PtrHashParams {
            remap: false,
            ..Default::default()
        };

        let sshash = SsHash::build(&[&text], minimizer, phf_builder);
        sshash.print_size();
    }

    fn pack(text: &[u8], packed: &mut Vec<u8>) {
        let mut packed_byte = 0;
        let mut packed_len = 0;
        for &base in text {
            packed_byte |= match base {
                b'a' | b'A' => 0,
                b'c' | b'C' => 1,
                b'g' | b'G' => 2,
                b't' | b'T' => 3,
                b'\r' | b'\n' => continue,
                _ => panic!(),
            } << (packed_len * 2);
            packed_len += 1;
            if packed_len == 4 {
                packed.push(packed_byte);
                packed_byte = 0;
                packed_len = 0;
            }
        }
    }
}
