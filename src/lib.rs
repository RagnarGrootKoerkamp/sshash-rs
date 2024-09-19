use itertools::{repeat_n, Itertools};
use minimizers::simd::packed::{IntoBpIterator, Packed};
use rayon::prelude::*;
use std::{cmp::Ordering, ops::Range};
use sux::{
    bit_field_vec,
    dict::EliasFano,
    traits::{BitFieldSlice, BitFieldSliceCore, BitFieldSliceMut, IndexedSeq},
};

pub trait PhfBuilder<T> {
    type Phf: Phf<T>;
    fn build(&self, keys: &[T]) -> Self::Phf;
}

pub trait Phf<T>: Sync {
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

pub trait Minimizer: Sync {
    fn minimizer_one(&self, window: impl IntoBpIterator) -> usize; // position of mini
    fn minimizers(&self, seq: impl IntoBpIterator) -> impl Iterator<Item = usize>; // absolute positions of minis
    fn k(&self) -> usize;
    fn w(&self) -> usize;
    fn l(&self) -> usize {
        self.k() + self.w() - 1
    }
}

pub trait BpStorage: Default + Sync {
    type BpSlice<'a>: IntoBpIterator
    where
        Self: 'a;

    fn push(&mut self, seq: Self::BpSlice<'_>) -> Range<usize>;
    fn concat<'a>(input_seqs: impl Iterator<Item = Self::BpSlice<'a>>) -> (Self, Vec<Range<usize>>)
    where
        Self: Sized + 'a;
    fn get(&self) -> Self::BpSlice<'_>;
    fn slice(&self, range: Range<usize>) -> Self::BpSlice<'_>;
    fn size(&self) -> usize;
    #[cfg(test)]
    fn random(n: usize, alphabet: usize) -> Self;
}

impl BpStorage for Vec<u8> {
    type BpSlice<'a> = &'a [u8];

    fn push(&mut self, seq: Self::BpSlice<'_>) -> Range<usize> {
        let start = seq.len();
        let end = start + IntoBpIterator::len(&seq);
        let range = start..end;
        self.extend(seq);
        range
    }

    fn concat<'a>(
        input_seqs: impl Iterator<Item = Self::BpSlice<'a>>,
    ) -> (Self, Vec<Range<usize>>) {
        let mut seq = vec![];
        let ranges = input_seqs
            .map(|slice| {
                let start = seq.len();
                let end = start + IntoBpIterator::len(&slice);
                let range = start..end;
                seq.extend(slice);
                range
            })
            .collect();
        (seq, ranges)
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

#[derive(Debug, Default)]
pub struct PackedVec {
    pub seq: Vec<u8>,
    pub len: usize,
}

impl BpStorage for PackedVec {
    type BpSlice<'a> = Packed<'a>;

    fn push<'a>(&mut self, seq: Self::BpSlice<'a>) -> Range<usize> {
        let start = 4 * self.seq.len() + seq.offset;
        let end = start + IntoBpIterator::len(&seq);
        let range = start..end;
        self.seq.extend(seq.seq);
        self.len = 4 * self.seq.len();
        range
    }
    fn concat<'a>(
        input_seqs: impl Iterator<Item = Self::BpSlice<'a>>,
    ) -> (Self, Vec<Range<usize>>) {
        let mut packed_vec = PackedVec {
            len: 0,
            seq: vec![],
        };
        let ranges = input_seqs.map(|slice| packed_vec.push(slice)).collect();
        (packed_vec, ranges)
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
    fn minimizer_one(&self, window: impl IntoBpIterator) -> usize {
        minimizers::simd::minimizer::minimizer_window_naive::<false>(window, self.k)
    }

    fn minimizers(&self, seq: impl IntoBpIterator) -> impl Iterator<Item = usize> {
        minimizers::simd::minimizer::minimizer_simd_it::<false>(seq, self.k, self.w)
            .map(|x| x as usize)
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
// TODO: External memory construction for offsets fs
pub struct SsHash<H: Phf<u64>, M: Minimizer, P: BpStorage> {
    minimizer: M,
    seqs: P,
    endpoints: Vec<usize>, // TODO EF?
    num_uniq_minis: usize,
    phf: H,
    sizes: sux::dict::elias_fano::EfSeq,
    offsets: sux::bits::BitFieldVec,
}

impl<H: Phf<u64>, M: Minimizer, P: BpStorage> SsHash<H, M, P> {
    // Convenience wrapper around build_from_slice that doesn't require type annotations.
    pub fn build<'a>(
        input_seqs: impl IntoIterator<Item = &'a P>,
        minimizer: M,
        phf_builder: impl PhfBuilder<u64, Phf = H>,
    ) -> Self
    where
        P: 'a,
    {
        Self::build_from_slices(
            input_seqs.into_iter().map(|x| x.get()),
            minimizer,
            phf_builder,
        )
    }
    pub fn build_from_slices<'a>(
        input_seqs: impl Iterator<Item = P::BpSlice<'a>>,
        minimizer: M,
        phf_builder: impl PhfBuilder<u64, Phf = H>,
    ) -> Self
    where
        P: 'a,
    {
        let (seq, ranges) = P::concat(input_seqs);

        Self::build_from_ranges(seq, ranges, minimizer, phf_builder)
    }

    fn build_from_ranges(
        seq: P,
        ranges: Vec<Range<usize>>,
        minimizer: M,
        phf_builder: impl PhfBuilder<u64, Phf = H>,
    ) -> SsHash<H, M, P> {
        let endpoints = ranges.iter().map(|x| x.end).collect_vec();
        let start = std::time::Instant::now();
        eprintln!("{:.1?}: minimizers..", start.elapsed());
        let mut minis = ranges
            .into_par_iter()
            .flat_map_iter(
                #[inline(always)]
                |range| {
                    minimizer
                        .minimizers(seq.slice(range.clone()))
                        .dedup()
                        .map(move |pos| pos + range.start)
                },
            )
            .map(|pos| {
                let val = seq.slice(pos..pos + minimizer.k()).to_word() as u64;
                (pos, val)
            })
            .collect::<Vec<_>>();
        eprintln!("{:.1?}: sort..", start.elapsed());
        minis.par_sort_by_key(|x| x.1);
        eprintln!("{:.1?}: uniq minis..", start.elapsed());
        // Todo: is passing an iterator to the phf builder sufficient?
        let uniq_mini_vals = minis.iter().map(|x| x.1).dedup().collect_vec();
        let num_uniq_minis = uniq_mini_vals.len();
        eprintln!("{:.1?}: build phf..", start.elapsed());
        let phf = phf_builder.build(&uniq_mini_vals);
        eprintln!("{:.1?}: collect ranges..", start.elapsed());

        let mut uniq_mini_range = {
            let threads = rayon::current_num_threads();
            let target_size = minis.len().div_ceil(threads);
            let mut start = 0;
            let ranges = (0..threads)
                .map(|_| {
                    let mut end = (start + target_size).min(minis.len());
                    if start >= minis.len() {
                        return minis.len()..minis.len();
                    }
                    let val = minis[end - 1].1;
                    while end < minis.len() && val == minis[end].1 {
                        end += 1;
                    }
                    let range = start..end;
                    start = end;
                    range
                })
                .collect_vec();

            ranges
                .into_par_iter()
                .flat_map_iter(|range| {
                    let r_start = range.start;
                    minis[range.clone()]
                        .iter()
                        .enumerate()
                        .chunk_by(|x| x.1 .1)
                        .into_iter()
                        .map(|(val, mut chunk)| {
                            let start = r_start + chunk.next().unwrap().0;
                            let end = start + 1 + chunk.count();
                            (phf.hash(val).unwrap(), start..end)
                        })
                        .collect_vec()
                })
                .collect::<Vec<_>>()
        };
        eprintln!("{:.1?}: sort by hash..", start.elapsed());
        uniq_mini_range.par_sort_by_key(|(idx, _range)| *idx);
        eprintln!("{:.1?}: fill sizes..", start.elapsed());
        let mut pos = 0;
        let mut sizes = uniq_mini_range
            .iter()
            .flat_map(|(idx, range)| {
                let extra = idx - pos;
                pos = *idx + 1;
                repeat_n(0, extra).chain(std::iter::once(range.len()))
            })
            .collect::<Vec<_>>();
        while sizes.len() <= phf.max() {
            sizes.push(0);
        }
        eprintln!("{:.1?}: accumulate sizes..", start.elapsed());
        sizes = sizes
            .into_iter()
            .scan(0, |acc, x| {
                let out = *acc;
                *acc += x;
                Some(out)
            })
            .collect_vec();

        eprintln!("{:.1?}: fill offsets..", start.elapsed());
        let offsets = uniq_mini_range
            .into_par_iter()
            .flat_map_iter(|(_idx, range)| minis[range].iter().map(|(pos, _val)| *pos))
            .collect::<Vec<_>>();
        assert_eq!(offsets.len(), minis.len());

        eprintln!("{:.1?}: packed offsets..", start.elapsed());
        let offset_bits = seq.get().len().ilog2() as usize + 1;
        let mut packed_offsets = bit_field_vec![offset_bits => 0; minis.len()];
        for (i, o) in offsets.into_iter().enumerate() {
            packed_offsets.set(i, o);
        }

        eprintln!("{:.1?}: Build sizes EF..", start.elapsed());
        let mut sizes_ef = sux::dict::EliasFanoBuilder::new(sizes.len(), *sizes.last().unwrap());
        for &size in &sizes {
            sizes_ef.push(size);
        }

        let sizes = sizes_ef.build_with_seq();

        eprintln!("{:.1?}: done.", start.elapsed());

        Self {
            minimizer,
            seqs: seq,
            endpoints,
            num_uniq_minis,
            phf,
            sizes,
            offsets: packed_offsets,
        }
    }

    pub fn query_one(&self, window: P::BpSlice<'_>) -> Option<(usize, usize)> {
        let minimizer_pos = self.minimizer.minimizer_one(window);
        let minimizer = window
            .sub_slice(minimizer_pos, self.minimizer.k())
            .to_word() as u64;
        let hash: usize = self.phf.hash(minimizer)?;
        let offsets_start = self.sizes.get(hash) as usize;
        let offsets_end = self.sizes.get(hash + 1) as usize;
        for idx in offsets_start..offsets_end {
            let offset = self.offsets.get(idx);
            if offset < minimizer_pos {
                continue;
            }
            let lmer_pos = offset as usize - minimizer_pos;
            if lmer_pos + self.minimizer.l() > self.seqs.get().len() {
                continue;
            }
            let lmer = &self.seqs.slice(lmer_pos..lmer_pos + self.minimizer.l());
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

    /// Returns (#found kmers, total # kmers)
    pub fn query_stream(&self, query: P::BpSlice<'_>) -> (usize, usize) {
        let l = self.minimizer.l();
        let minimizer_it = self.minimizer.minimizers(query);
        let mut found = 0;
        let mut last_minimizer_pos = usize::MAX;
        let mut last_lmer_pos = None;
        let mut offsets = 0..0;
        for (i, minimizer_pos) in minimizer_it.enumerate() {
            let same_minimizer = minimizer_pos == last_minimizer_pos;
            last_minimizer_pos = minimizer_pos;

            // Update the range of offsets for the current minimizer if it changed.
            if !same_minimizer {
                let minimizer = query.sub_slice(minimizer_pos, self.minimizer.k()).to_word() as u64;

                let hash: usize = self.phf.hash(minimizer).unwrap();
                let offsets_start = self.sizes.get(hash) as usize;
                let offsets_end = self.sizes.get(hash + 1) as usize;
                offsets = offsets_start..offsets_end;
            }

            // Try extending the previous match, regardless of where the current minimizer is.
            // TODO: Check we don't cross unitig ends.
            if let Some(last_pos) = last_lmer_pos {
                let candidate_lmer_pos = last_pos + 1;
                if candidate_lmer_pos + l <= self.seqs.get().len()
                    && query.sub_slice(i + l - 1, 1).to_word()
                        == self
                            .seqs
                            .slice(candidate_lmer_pos + l - 1..candidate_lmer_pos + l)
                            .to_word()
                {
                    found += 1;
                    last_lmer_pos = Some(candidate_lmer_pos);
                    continue;
                }
            }

            // Iterate over all occurrences of the minimizer.
            'found: {
                let relative_minimizer_pos = minimizer_pos - i;
                for idx in offsets.clone() {
                    let offset = self.offsets.get(idx);
                    if offset < relative_minimizer_pos {
                        continue;
                    }
                    let lmer_pos = offset as usize - relative_minimizer_pos;
                    if lmer_pos + self.minimizer.l() > self.seqs.get().len() {
                        continue;
                    }
                    let lmer = self.seqs.slice(lmer_pos..lmer_pos + self.minimizer.l());
                    let sequence_lmer = lmer.to_word();
                    let query_lmer = query.sub_slice(i, l).to_word();
                    if sequence_lmer == query_lmer {
                        found += 1;
                        last_lmer_pos = Some(lmer_pos);
                        break 'found;
                    }
                }
                // not found
                last_lmer_pos = None;
            }
        }
        (found, query.len() - self.minimizer.l() + 1)
    }

    pub fn print_size(&self) {
        use size::Size;
        use std::mem::size_of_val;
        // All sizes in bytes.
        let seqs = self.seqs.size();
        let endpoints = size_of_val(self.endpoints.as_slice());
        let phf = self.phf.size();
        // TODO: Include size of rank-select structures.
        let sizes = EliasFano::<(), ()>::estimate_size(self.offsets.len(), self.sizes.len()) / 8;
        let offsets = self.offsets.len() * self.offsets.bit_width() / 8;
        let total = seqs + endpoints + phf + sizes + offsets;

        let num_bp = self.seqs.get().len() as f32;
        let num_minis = self.offsets.len() as f32;

        eprintln!(
            "part       {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "len", "max", "memory", "bits/kmer", "bits/mini", "bits/uniq mini"
        );
        eprintln!(
            "seqs:      {:>10} {:>10} {:>10} {:>10.1} {:>10.1} {:>10.1}",
            self.seqs.get().len(),
            3,
            format!("{}", Size::from_bytes(seqs)),
            8. * seqs as f32 / num_bp,
            8. * seqs as f32 / num_minis,
            8. * seqs as f32 / self.num_uniq_minis as f32
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
            self.seqs.get().len(),
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

    #[test]
    fn build() {
        let minimizer = NtMinimizer { k: 19, w: 11 };
        let phf_builder = ptr_hash::PtrHashParams {
            remap: false,
            ..Default::default()
        };
        let n = 1000000;
        let num_seqs = 20;
        let seqs = (0..num_seqs).map(|_| PackedVec::random(n, 4)).collect_vec();
        let start = std::time::Instant::now();
        let _sshash = SsHash::build(&seqs, minimizer, phf_builder);
        eprintln!("Construction took {:?}", start.elapsed());
    }

    fn test_pos<P: BpStorage>(
        minimizer: impl Minimizer,
        phf_builder: impl PhfBuilder<u64>,
        alphabet: usize,
    ) {
        let n = 100000;
        let seq = P::random(n, alphabet);
        let l = minimizer.k() + minimizer.w() - 1;
        let sshash = SsHash::build([&seq], minimizer, phf_builder);

        // The alphabet is large enough that we assume no duplicate k-mers occur.
        for i in 0..seq.get().len() - l {
            let window = seq.slice(i..i + l);
            assert_eq!(sshash.query_one(window), Some((i, 0)));
        }
    }

    fn test_neg<P: BpStorage + std::fmt::Debug>(
        minimizer: impl Minimizer,
        phf_builder: impl PhfBuilder<u64>,
        alphabet: usize,
    ) {
        let n = 100000;
        let seq = P::random(n, alphabet);
        let l = minimizer.l();
        let sshash = SsHash::build([&seq], minimizer, phf_builder);

        // The alphabet is large enough that we assume no duplicate k-mers occur.
        for _ in 0..seq.get().len() {
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

    fn test_pos_stream<P: BpStorage>(
        minimizer: impl Minimizer,
        phf_builder: impl PhfBuilder<u64>,
        alphabet: usize,
    ) {
        let n = 1000;
        let l = minimizer.l();
        let seq = P::random(n, alphabet);
        let sshash = SsHash::build([&seq], minimizer, phf_builder);

        assert_eq!(sshash.query_stream(seq.get()), (n - l + 1, n - l + 1));
    }

    fn test_neg_stream<P: BpStorage + std::fmt::Debug>(
        minimizer: impl Minimizer,
        phf_builder: impl PhfBuilder<u64>,
        alphabet: usize,
    ) {
        let n = 100000;
        let l = minimizer.l();
        let seq = P::random(n, alphabet);
        let sshash = SsHash::build([&seq], minimizer, phf_builder);

        let query = P::random(n, alphabet);
        assert_eq!(sshash.query_stream(query.get()), (0, n - l + 1));
    }

    #[test]
    fn ntmini_ptrhash_pos_packed_stream() {
        let minimizer = NtMinimizer { k: 19, w: 11 };
        let phf_builder = ptr_hash::PtrHashParams {
            remap: false,
            ..Default::default()
        };
        test_pos_stream::<PackedVec>(minimizer, phf_builder, 4);
    }

    #[test]
    fn ntmini_ptrhash_neg_packed_stream() {
        let minimizer = NtMinimizer { k: 19, w: 11 };
        let phf_builder = ptr_hash::PtrHashParams {
            remap: false,
            ..Default::default()
        };
        test_neg_stream::<PackedVec>(minimizer, phf_builder, 4);
    }

    #[ignore]
    #[test]
    fn print_size() {
        let seq = PackedVec::random(10000000, 4);
        let k = 21;
        let w = 11;
        let minimizer = NtMinimizer { k, w };
        let phf_builder = ptr_hash::PtrHashParams {
            remap: false,
            ..Default::default()
        };

        let sshash = SsHash::build([&seq], minimizer, phf_builder);
        sshash.print_size();
    }

    #[ignore]
    #[test]
    fn print_size_hg() {
        eprintln!("Reading..");
        let start = std::time::Instant::now();
        let Ok(mut reader) = needletail::parse_fastx_file("human-genome.fa") else {
            panic!("Did not find human-genome.fa. Add/symlink it to test runtime on it.");
        };
        let mut seq = PackedVec::default();
        let mut ranges = vec![];
        let mut pack_buf = vec![];
        while let Some(r) = reader.next() {
            let r = r.unwrap();
            let len = pack(&r.raw_seq(), &mut pack_buf);
            let packed = Packed {
                seq: &pack_buf,
                offset: 0,
                len,
            };
            ranges.push(seq.push(packed));
            pack_buf.clear();
        }
        eprintln!("Packing took {:?}", start.elapsed());

        let k = 20;
        let w = 12;
        let minimizer = NtMinimizer { k, w };
        let phf_builder = ptr_hash::PtrHashParams {
            remap: false,
            ..Default::default()
        };

        let sshash = SsHash::build_from_ranges(seq, ranges, minimizer, phf_builder);
        sshash.print_size();
    }

    #[test]
    fn test_pack() {
        for n in 0..=128 {
            let seq = (0..n)
                .map(|_| b"ACGT"[rand::random::<u8>() as usize % 4])
                .collect_vec();
            let mut packed_1 = vec![];
            let len1 = naive::pack(&seq, &mut packed_1);
            let mut packed_2 = vec![];
            let len2 = pack(&seq, &mut packed_2);
            assert_eq!(len1, len2);
            assert_eq!(packed_1, packed_2);
        }
    }

    fn pack(seq: &[u8], packed: &mut Vec<u8>) -> usize {
        let mut packed_len = 0;
        let last = seq.len() / 8 * 8;
        for i in (0..last).step_by(8) {
            let chunk = &seq[i..i + 8].try_into().unwrap();
            let word = u64::from_ne_bytes(*chunk);
            let packed_byte = unsafe { std::arch::x86_64::_pext_u64(word, 0x0606060606060606) };
            packed.push(packed_byte as u8);
            packed.push((packed_byte >> 8) as u8);
            packed_len += 8;
        }

        let mut packed_byte = 0;
        for &base in &seq[last..] {
            packed_byte |= match base {
                b'a' | b'A' => 0,
                b'c' | b'C' => 1,
                b'g' | b'G' => 3,
                b't' | b'T' => 2,
                b'\r' | b'\n' => continue,
                _ => panic!(),
            } << (packed_len * 2);
            packed_len += 1;
            if packed_len % 4 == 0 {
                packed.push(packed_byte);
                packed_byte = 0;
            }
        }
        if packed_len % 4 != 0 {
            packed.push(packed_byte);
        }
        packed_len
    }
}
