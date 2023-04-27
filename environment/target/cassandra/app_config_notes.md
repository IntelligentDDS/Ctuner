* JVM.options
  * Heap size
  * Young generation size
  * #-XX:+UseG1GC
  * #-XX:G1RSetUpdatingPauseTimePercent=5
  * #-XX:MaxGCPauseMillis=500
  * ParallelGCThreads & ConcGCThreads
  *
* cassandra.yaml
  * trickle_fsync
  * trickle_fsync_interval_in_kb
  * rpc_server_type
  * rpc_send_buff_size_in_bytes
  * rpc_recv_buff_size_in_bytes
  * thrift_framed_transport_size_in_mb
  * concurrent_compactors
  * compaction_throughput_mb_per_sec
* init.cql
  * see http://thelastpickle.com/blog/2018/08/08/compression_performance.html
  *
