use itertools::Itertools;
use std::cmp::Ordering;

pub trait PhfBuilder<T> {
    type Phf: Phf<T>;
    fn build(&self, keys: &[T]) -> Self::Phf;
}

pub trait Phf<T> {
    fn hash(&self, key: T) -> Option<usize>; // hash
    fn max(&self) -> usize; // upper bound on hashes
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
    phf: H,
    sizes: Vec<usize>,   // Todo EF
    offsets: Vec<usize>, // Todo CompactVec
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

        let mut offsets = vec![0; minis.len()];
        for (mini_val, chunk) in minis.iter().chunk_by(|x| x.1).into_iter() {
            if let Some(hash) = phf.hash(mini_val) {
                let start = sizes[hash] as usize;
                for (i, (pos, _)) in chunk.enumerate() {
                    offsets[start + i] = *pos;
                }
            }
        }

        Self {
            minimizer,
            text,
            endpoints,
            phf,
            sizes,
            offsets,
        }
    }

    pub fn query_one(&self, window: &[u8]) -> Option<(usize, usize)> {
        let (minimizer_pos, minimizer) = self.minimizer.minimizer_one(window);
        let hash: usize = self.phf.hash(minimizer)?;
        let offsets_start = self.sizes[hash] as usize;
        let offsets_end = self.sizes[hash + 1] as usize;
        let offsets = &self.offsets[offsets_start..offsets_end];
        for offset in offsets {
            let lmer_pos = *offset as usize - minimizer_pos;
            let lmer = &self.text[lmer_pos..lmer_pos + self.minimizer.l()];
            if lmer == window {
                return Some((
                    lmer_pos,
                    self.endpoints
                        .binary_search_by(|x| {
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
}
