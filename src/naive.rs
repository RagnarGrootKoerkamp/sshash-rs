use super::*;
use std::collections::HashMap;
use std::hash::Hash;

pub struct NaivePhfBuilder<T> {
    _marker: std::marker::PhantomData<T>,
}

impl<T> NaivePhfBuilder<T> {
    pub fn new() -> Self {
        NaivePhfBuilder {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Hash + Eq + Copy + Sync> PhfBuilder<T> for NaivePhfBuilder<T> {
    type Phf = NaivePhf<T>;

    fn build(&self, keys: &[T]) -> Self::Phf {
        NaivePhf {
            hashmap: keys.iter().enumerate().map(|(i, key)| (*key, i)).collect(),
        }
    }
}

pub struct NaivePhf<T> {
    hashmap: HashMap<T, usize>,
}

impl<T: Hash + Eq + Sync> Phf<T> for NaivePhf<T> {
    fn hash(&self, key: T) -> Option<usize> {
        self.hashmap.get(&key).copied()
    }

    fn max(&self) -> usize {
        self.hashmap.len()
    }

    fn size(&self) -> usize {
        self.hashmap.capacity() * (std::mem::size_of::<T>() + std::mem::size_of::<usize>())
    }
}

#[derive(Epserde, Copy, Clone)]
#[repr(C)]
#[zero_copy]
pub struct NaiveMinimizer {
    pub k: usize,
    pub w: usize,
}

impl Minimizer for NaiveMinimizer {
    fn minimizer_one(&self, window: impl Seq) -> usize {
        assert_eq!(window.len(), self.l());
        (0..self.w)
            .map(|i| {
                let kmer = window.slice(i..i + self.k).to_word();
                let hash = fxhash::hash64(&kmer);
                (i, hash)
            })
            .min_by_key(|&(_, hash)| hash)
            .unwrap()
            .0
    }

    fn minimizers(&self, text: impl Seq) -> impl Iterator<Item = usize> {
        (0..text.len() - self.l() + 1).map(move |i| {
            let window = text.slice(i..i + self.l());
            i + self.minimizer_one(window)
        })
    }

    fn k(&self) -> usize {
        self.k
    }

    fn w(&self) -> usize {
        self.w
    }
}
