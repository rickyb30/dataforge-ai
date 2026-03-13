-- DataForge SQL module test data
-- Run: psql -h localhost -U postgres -d testdb -f seed_postgres.sql

DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

CREATE TABLE customers (
    customer_id   SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    email         VARCHAR(150),
    region        VARCHAR(50),
    created_at    DATE DEFAULT CURRENT_DATE
);

CREATE TABLE orders (
    order_id      SERIAL PRIMARY KEY,
    customer_id   INT REFERENCES customers(customer_id),
    amount        NUMERIC(12,2) NOT NULL,
    product       VARCHAR(100),
    order_date    DATE NOT NULL
);

INSERT INTO customers (name, email, region) VALUES
('Alice Chen',     'alice@example.com',   'NA'),
('Bob Martinez',   'bob@example.com',     'NA'),
('Carol Davis',    'carol@example.com',   'EMEA'),
('Dan Wilson',     'dan@example.com',     'APAC'),
('Eve Johnson',    'eve@example.com',     'NA');

INSERT INTO orders (customer_id, amount, product, order_date) VALUES
(1, 500.00,  'Widget A',  '2025-12-01'),
(1, 750.00,  'Widget B',  '2025-12-15'),
(2, 1200.00, 'Widget A',  '2025-11-20'),
(2, 300.00,  'Widget C',  '2026-01-05'),
(3, 2500.00, 'Widget B',  '2026-01-10'),
(3, 150.00,  'Widget A',  '2025-10-01'),
(4, 800.00,  'Widget C',  '2026-02-01'),
(4, 1100.00, 'Widget A',  '2026-02-10'),
(5, 450.00,  'Widget B',  '2025-09-15'),
(5, 3000.00, 'Widget A',  '2026-01-20'),
(1, 200.00,  'Widget C',  '2026-02-20'),
(2, 950.00,  'Widget B',  '2026-02-22');
