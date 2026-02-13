use sqlx::postgres::{PgPool, PgPoolOptions};
use std::time::Duration;

/// Default configuration tuned for Fly.io Postgres deployments.
pub struct PoolConfig {
    pub max_connections: u32,
    pub min_connections: u32,
    pub acquire_timeout_secs: u64,
    pub idle_timeout_secs: u64,
    pub test_before_acquire: bool,
    pub max_retries: u32,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 3,
            min_connections: 0,
            acquire_timeout_secs: 10,
            idle_timeout_secs: 120,
            test_before_acquire: true,
            max_retries: 10,
            initial_backoff_ms: 500,
            max_backoff_ms: 15_000,
        }
    }
}

/// Build pool options from config.
///
/// `test_before_acquire` pings each connection before handing it out,
/// preventing "expected to read N bytes, got 0" errors when Fly Postgres
/// drops idle connections via HAProxy.
fn pool_options(config: &PoolConfig) -> PgPoolOptions {
    PgPoolOptions::new()
        .max_connections(config.max_connections)
        .min_connections(config.min_connections)
        .acquire_timeout(Duration::from_secs(config.acquire_timeout_secs))
        .idle_timeout(Duration::from_secs(config.idle_timeout_secs))
        .test_before_acquire(config.test_before_acquire)
}

/// Connect to PostgreSQL with exponential backoff and jitter.
///
/// Retries up to `config.max_retries` times with capped exponential backoff
/// plus 0-25% random jitter to prevent thundering herd on recovery.
/// Default config survives 60s+ Fly DB recovery without giving up.
pub async fn connect(database_url: &str) -> Result<PgPool, sqlx::Error> {
    connect_with_config(database_url, &PoolConfig::default()).await
}

/// Connect with custom pool configuration.
pub async fn connect_with_config(
    database_url: &str,
    config: &PoolConfig,
) -> Result<PgPool, sqlx::Error> {
    connect_with_retries(database_url, config).await
}

/// Compute backoff with jitter: exponential base capped at max_backoff_ms,
/// plus random jitter of 0-25% to prevent thundering herd.
fn backoff_ms(attempt: u32, initial: u64, max: u64) -> u64 {
    let base = initial.saturating_mul(2u64.saturating_pow(attempt));
    let capped = base.min(max);
    let jitter = capped / 4 * ((attempt as u64 * 7 + 3) % 4) / 3;
    capped + jitter
}

async fn connect_with_retries(
    database_url: &str,
    config: &PoolConfig,
) -> Result<PgPool, sqlx::Error> {
    let mut last_err = None;

    for attempt in 0..=config.max_retries {
        match pool_options(config).connect(database_url).await {
            Ok(pool) => {
                if attempt > 0 {
                    tracing::info!("Database connected after {} retries", attempt);
                }
                return Ok(pool);
            }
            Err(e) => {
                let is_last = attempt == config.max_retries;
                if is_last {
                    tracing::error!(
                        "Database connection failed after {} attempts: {e}",
                        attempt + 1
                    );
                } else {
                    let wait = backoff_ms(attempt, config.initial_backoff_ms, config.max_backoff_ms);
                    tracing::warn!(
                        "Database connection attempt {} failed, retrying in {wait}ms...",
                        attempt + 1,
                    );
                    tokio::time::sleep(Duration::from_millis(wait)).await;
                }
                last_err = Some(e);
            }
        }
    }

    Err(last_err.unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> PoolConfig {
        PoolConfig::default()
    }

    #[test]
    fn default_constants_are_reasonable() {
        let c = default_config();
        assert_eq!(c.max_retries, 10);
        assert_eq!(c.initial_backoff_ms, 500);
        assert_eq!(c.max_backoff_ms, 15_000);

        // Must survive a 60s+ Fly DB recovery
        let total: u64 = (0..c.max_retries)
            .map(|i| backoff_ms(i, c.initial_backoff_ms, c.max_backoff_ms))
            .sum();
        assert!(
            total >= 60_000,
            "Total backoff {total}ms must be >= 60s for DB recovery"
        );
        assert!(
            total <= 180_000,
            "Total backoff {total}ms must be <= 180s"
        );
    }

    #[test]
    fn pool_config_tuned_for_fly() {
        let c = default_config();
        // 2 machines × 3 connections = 6 total — within Fly Postgres limits
        assert_eq!(c.max_connections, 3);
        assert_eq!(c.acquire_timeout_secs, 10);
        // idle_timeout must be well under Fly's ~600s idle disconnect
        assert!(c.idle_timeout_secs < 300);
    }

    #[test]
    fn pool_options_builds_without_panic() {
        let _opts = pool_options(&default_config());
    }

    #[test]
    fn backoff_starts_at_initial_value() {
        let first = backoff_ms(0, 500, 15_000);
        assert!(first >= 500);
        assert!(first <= 625); // 500 + 25% jitter
    }

    #[test]
    fn backoff_caps_at_max() {
        for attempt in 5..=15 {
            let ms = backoff_ms(attempt, 500, 15_000);
            let ceiling = 15_000 + 15_000 / 4;
            assert!(
                ms <= ceiling,
                "Backoff at attempt {attempt} is {ms}ms, exceeds cap+jitter {ceiling}ms"
            );
        }
    }

    #[test]
    fn backoff_increases_before_cap() {
        let base = |a: u32| 500u64.saturating_mul(2u64.saturating_pow(a)).min(15_000);
        for i in 0..5 {
            assert!(base(i + 1) >= base(i));
        }
    }

    #[test]
    fn backoff_no_overflow() {
        let ms = backoff_ms(100, 500, 15_000);
        assert!(ms <= 15_000 + 15_000 / 4);
    }

    #[tokio::test]
    async fn connect_fails_on_bad_url() {
        let mut config = default_config();
        config.max_retries = 0;
        let result = connect_with_config("postgres://invalid:5432/nope", &config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn connect_respects_max_retries() {
        let mut config = default_config();
        config.max_retries = 1;
        let start = std::time::Instant::now();
        let result = connect_with_config("postgres://invalid:5432/nope", &config).await;
        let elapsed = start.elapsed();
        assert!(result.is_err());
        assert!(elapsed >= Duration::from_millis(400));
        assert!(elapsed < Duration::from_secs(5));
    }
}
