test:
    cargo test -r

perf:
    cargo build -r --tests
    perf record cargo test -r build -- --nocapture
    # perf report
stat:
    cargo build -r --tests
    perf stat cargo test -r build -- --nocapture
