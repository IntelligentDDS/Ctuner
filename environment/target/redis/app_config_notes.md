* redis.conf.j2 => redis.conf
 * long_time long_time_changedkeys
 * medium_time medium_time_changedkeys
 * short_time short_time_changedkeys
  save 900 1
  save 300 10
  save 60 10000
 * appendonly_switch
 * append_fsync
 * no_appendfsync_on_rewrite_switch

 * maxmemory
 * # maxmemory-policy noeviction 暂时未添加
