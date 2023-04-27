create keyspace ycsb  WITH REPLICATION = {'class' : 'SimpleStrategy', 'replication_factor': 1 };
USE ycsb;
DROP TABLE IF EXISTS usertable;
create table usertable (
    y_id varchar primary key,
    field0 varchar,
    field1 varchar,
    field2 varchar,
    field3 varchar,
    field4 varchar,
    field5 varchar,
    field6 varchar,
    field7 varchar,
    field8 varchar,
    field9 varchar)
WITH compression = {
    'sstable_compression': 'LZ4Compressor',
    'chunk_length_kb': {{2 ** app_config.compression_chunk_length}}
};
