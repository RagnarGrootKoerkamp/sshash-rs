use itertools::Itertools;
use std::cmp::Ordering;
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

pub trait Minimizer {
    fn minimizer_one(&self, window: &[u8]) -> (usize, u64); // (pos, value)
    fn minimizers(&self, text: &[u8]) -> impl Iterator<Item = (usize, u64)>; // (pos, value)
    fn k(&self) -> usize;
    fn w(&self) -> usize;
    fn l(&self) -> usize {
        self.k() + self.w() - 1
    }
}

pub struct SsHash<H: Phf<u64>, M: Minimizer> {
    minimizer: M,
    text: Vec<u8>,         // Todo bitpacked
    endpoints: Vec<usize>, // Todo EF
    num_uniq_minis: usize,
    phf: H,
    sizes: sux::dict::EliasFano<SelectAdaptConst<sux::bits::BitVec<Box<[usize]>>>>,
    offsets: sux::bits::BitFieldVec,
}

impl<H: Phf<u64>, M: Minimizer> SsHash<H, M> {
    pub fn build(text: &[&[u8]], minimizer: M, phf_builder: impl PhfBuilder<u64, Phf = H>) -> Self {
        let endpoints = text
            .iter()
            .scan(0, |acc, x| {
                *acc += x.len();
                Some(*acc)
            })
            .collect_vec();
        let text = text.concat();

        // Todo split at input seq boundaries
        let mut minis = minimizer.minimizers(&text).dedup().collect_vec();
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

        let offset_bits = text.len().ilog2() as usize + 1;
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

    pub fn print_size(&self) {
        use size::Size;
        use std::mem::size_of_val;
        // All sizes in bytes.
        let text = size_of_val(self.text.as_slice());
        let endpoints = size_of_val(self.endpoints.as_slice());
        let phf = self.phf.size();
        // TODO: Include size of rank-select structures.
        let sizes = EliasFano::<(), ()>::estimate_size(self.offsets.len(), self.sizes.len()) / 8;
        let offsets = self.offsets.len() * self.offsets.bit_width() / 8;
        let total = text + endpoints + phf + sizes + offsets;

        let num_bp = self.text.len() as f32;
        let num_minis = self.offsets.len() as f32;

        eprintln!(
            "part       {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "len", "max", "memory", "bits/kmer", "bits/mini", "bits/uniq mini"
        );
        eprintln!(
            "text:      {:>10} {:>10} {:>10} {:>10.1} {:>10.1} {:>10.1}",
            self.text.len(),
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
            self.text.len(),
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

    pub fn query_one(&self, window: &[u8]) -> Option<(usize, usize)> {
        let (minimizer_pos, minimizer) = self.minimizer.minimizer_one(window);
        let hash: usize = self.phf.hash(minimizer)?;
        let offsets_start = self.sizes.get(hash) as usize;
        let offsets_end = self.sizes.get(hash + 1) as usize;
        for idx in offsets_start..offsets_end {
            let offset = self.offsets.get(idx);
            let lmer_pos = offset as usize - minimizer_pos;
            let lmer = &self.text[lmer_pos..lmer_pos + self.minimizer.l()];
            if lmer == window {
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
}

#[cfg(test)]
mod naive;

#[cfg(test)]
mod test {
    use super::*;
    use naive::*;

    #[test]
    fn positive_queries() {
        let text = (0..100000).map(|_| rand::random::<u8>()).collect_vec();
        let k = 21;
        let w = 11;
        let l = 31;
        let minimizer = NaiveMinimizer { k, w };
        let phf_builder = NaivePhfBuilder::new();
        let sshash = SsHash::build(&[&text], minimizer, phf_builder);

        // The alphabet is large enough that we assume no duplicate k-mers occur.
        for (i, window) in text.windows(l).enumerate() {
            assert_eq!(sshash.query_one(window), Some((i, 0)));
        }
    }

    #[test]
    fn negative_queries() {
        let text = (0..100000).map(|_| rand::random::<u8>()).collect_vec();
        let k = 21;
        let w = 11;
        let l = 31;
        let minimizer = NaiveMinimizer { k, w };
        let phf_builder = NaivePhfBuilder::new();
        let sshash = SsHash::build(&[&text], minimizer, phf_builder);

        for _ in 0..text.len() {
            let window = (0..l).map(|_| rand::random::<u8>()).collect_vec();
            assert_eq!(sshash.query_one(&window), None);
        }
    }

    #[ignore]
    #[test]
    fn print_size() {
        let text = (0..1000000).map(|_| rand::random::<u8>()).collect_vec();
        let k = 21;
        let w = 11;
        let minimizer = NaiveMinimizer { k, w };
        let phf_builder = NaivePhfBuilder::new();
        let sshash = SsHash::build(&[&text], minimizer, phf_builder);
        sshash.print_size();
    }
}
