CREATE TABLE simple_table (
    id INT NOT NULL,
    name VARCHAR(50) NOT NULL,
    note TEXT,
    CONSTRAINT simple_table_pk PRIMARY KEY (id)
);

CREATE TABLE constrained_table (
    id INT NOT NULL,
    simple_id INT NOT NULL,
    code VARCHAR(10) NOT NULL,
    small_val SMALLINT,
    big_val BIGINT,
    price DECIMAL(10, 2),
    generic_amount DECIMAL,
    created_at DATETIME,
    CONSTRAINT constrained_table_pk PRIMARY KEY (id),
    CONSTRAINT constrained_table_uk UNIQUE (code),
    CONSTRAINT constrained_table_ck CHECK (small_val > 0),
    CONSTRAINT constrained_table_fk FOREIGN KEY (simple_id) REFERENCES simple_table (id) ON DELETE CASCADE
);

CREATE INDEX ix_constrained_table_code ON constrained_table (code);
CREATE UNIQUE INDEX ux_constrained_table_big_val ON constrained_table (big_val);

CREATE TABLE sqlite_stat1 (tbl TEXT, idx TEXT, stat TEXT);
