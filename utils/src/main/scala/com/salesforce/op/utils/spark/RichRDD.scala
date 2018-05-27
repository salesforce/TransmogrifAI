/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.utils.spark

import com.twitter.algebird.Semigroup
import org.apache.hadoop.io.compress.CompressionCodec
import org.apache.hadoop.io.{NullWritable, Text}
import org.apache.hadoop.mapred.{JobConf, OutputFormat, TextOutputFormat}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag


object RichRDD {

  /**
   * An enhanced RDD with more general join methods.
   *
   * @param rdd
   * @tparam U
   */
  implicit class RichRDD[U: ClassTag](rdd: RDD[U]) {

    private def transform[T, K](rdd: RDD[T], key: T => K) = rdd.keyBy(key)

    /**
     * Splits this RDD in two RDDs according to a predicate.
     *
     * @param p the predicate on which to split by.
     * @return a pair of RDDs: the RDD that satisfies the predicate `p` and the RDD that does not.
     */
    def split(p: U => Boolean): (RDD[U], RDD[U]) = {
      val splits = rdd.mapPartitions { iter =>
        val (left, right) = iter.partition(p)
        (left :: right :: Nil).iterator
      }
      val left = splits.mapPartitions(_.next())
      val right = splits.mapPartitions(it => {
        it.next() // skip one
        it.next()
      })
      (left, right)
    }

    /**
     * A more general method to do full outer joins.
     * Allows you to specify the transformations on the RDDs that will produce the join keys
     *
     * @param rdd2 rdd to join with
     * @param key1 transform function to generate key for first rdd
     * @param key2 transform function to generate key for second rdd
     * @tparam K
     * @tparam V
     * @return rdd1 fullOuterJoin rdd2
     */
    def outerJoinBy[K: ClassTag, V: ClassTag](rdd2: RDD[V])
      (key1: U => K, key2: V => K): RDD[(K, (Option[U], Option[V]))] = {
      transform(rdd, key1).fullOuterJoin(transform(rdd2, key2))
    }


    /**
     * A more general method to do left joins.
     * Allows you to specify the transformations on the RDDs that will produce the join keys
     *
     * @param rdd2 rdd to join with
     * @param key1 transform function to generate key for first rdd
     * @param key2 transform function to generate key for second rdd
     * @tparam K
     * @tparam V
     * @return rdd1 leftOuterJoin rdd2
     */
    def leftJoinBy[K: ClassTag, V: ClassTag](rdd2: RDD[V])(key1: U => K, key2: V => K): RDD[(K, (U, Option[V]))] = {
      transform(rdd, key1).leftOuterJoin(transform(rdd2, key2))
    }

    /**
     * A more general method to do left outer joins.
     * Allows you to specify the transformations on the RDDs that will produce the join keys
     *
     * @param rdd2 rdd to join with
     * @param key1 transform function to generate key for first rdd
     * @param key2 transform function to generate key for second rdd
     * @tparam K
     * @tparam V
     * @return rdd1 join rdd2
     */
    def joinBy[K: ClassTag, V: ClassTag](rdd2: RDD[V])(key1: U => K, key2: V => K): RDD[(K, (U, V))] = {
      transform(rdd, key1).join(transform(rdd2, key2))
    }


    /**
     * A more general method to do cogroup.
     * Allows you to specify the transformations on the RDDs that will produce the join keys
     *
     * @param rdd2 rdd to join with
     * @param key1 transform function to generate key for first rdd
     * @param key2 transform function to generate key for second rdd
     * @tparam K
     * @tparam V
     * @return rdd1 cogroup rdd2
     */
    def cogroupBy[K: ClassTag, V: ClassTag](rdd2: RDD[V])
      (key1: U => K, key2: V => K): RDD[(K, (Iterable[U], Iterable[V]))] = {
      transform(rdd, key1).cogroup(transform(rdd2, key2))
    }

    /**
     * Map over the RDD with an long accumulator
     *
     * @param f           map function
     * @param counterName accumulator name
     * @tparam T
     * @return rdd.map(f)
     */
    def mapWithCount[T: ClassTag](f: U => T, counterName: String): RDD[T] = {
      val counter = rdd.sparkContext.longAccumulator(counterName)
      rdd.map { x => counter.add(1L); f(x) }
    }

    /**
     * A more efficient implementation of groupBy and then reduceByKey
     *
     * @param keyFn    function of which to group by
     * @param valueFn  function of which to generate the value from
     * @param reduceFn combiner function
     * @tparam K
     * @tparam V
     * @return rdd of keys and the reduced values
     */
    def groupAndReduce[K: ClassTag, V: ClassTag](keyFn: U => K, valueFn: U => V, reduceFn: (V, V) => V): RDD[(K, V)] = {
      rdd
        .map(r => (keyFn(r), valueFn(r)))
        .reduceByKey(reduceFn)
    }

    /**
     * An efficient implementation of count by key
     *
     * @param keyFn function of which to group by
     * @tparam K
     * @return rdd of keys and their counts
     */
    def groupAndCount[K: ClassTag](keyFn: U => K): RDD[(K, Long)] =
      groupAndSum[K, Long](keyFn, _ => 1L)

    /**
     * An efficient implementation of sum by key
     *
     * @param keyFn   function of which to group by
     * @param valueFn function of which to generate the value from
     * @tparam K
     * @tparam V
     * @return rdd of keys their sums
     */
    def groupAndSum[K, V](keyFn: U => K, valueFn: U => V)
      (implicit ctk: ClassTag[K], ctv: ClassTag[V], sg: Semigroup[V]): RDD[(K, V)] =
      groupAndReduce(keyFn, valueFn, sg.plus)

    /**
     * Repartition to the min of maxPartitionCount and the previous number of partitions.
     * Usually you would want maxPartitionCount to be 200 to get around the 200 bug
     *
     * @param maxPartitionCount max number of partitions (default 200)
     * @return shuffled rdd with same number of partitions or maxPartitionCount, whichever is less
     */
    def repartitionToMaxPartitions(maxPartitionCount: Int = 200): RDD[U] =
      rdd.repartition(math.min(rdd.partitions.length, maxPartitionCount))


    /**
     * Repartition to the min of floor(rdd.count / recordsPerPartition) + 1 and the previous number of partitions
     * Attention: involves the execution of rdd.count() operation
     *
     * @param recordsPerPartition number of records per partition
     * @return rdd with same number of partitions or floor(rdd.count / recordsPerPartition) + 1 partitions,
     *         whichever is less
     */
    def repartitionByRecords(recordsPerPartition: Int): RDD[U] = {
      val count = rdd.count().toDouble
      val partitions = math.floor(count / recordsPerPartition).toInt + 1
      rdd.coalesce(partitions, shuffle = false)
    }

    /**
     * Output the RDD to any Hadoop-supported file system, using a Hadoop `OutputFormat` class
     * supporting the key and value types K and V in this RDD.
     *
     * @note We should make sure our tasks are idempotent when speculation is enabled, i.e. do
     *       not use output committer that writes data directly.
     *       There is an example in https://issues.apache.org/jira/browse/SPARK-10063 to show the bad
     *       result of using direct output committer with speculation enabled.
     *
     * @param path    path to write the data
     * @param codec   optional codec to use
     * @param jobConf optional hadoop configuration
     */
    def saveAsTextFile(
      path: String,
      codec: Option[Class[_ <: CompressionCodec]],
      jobConf: JobConf
    ): Unit = {
      val nullWritableClassTag = implicitly[ClassTag[NullWritable]]
      val textClassTag = implicitly[ClassTag[Text]]
      val outputFormatClassTag = implicitly[ClassTag[TextOutputFormat[NullWritable, Text]]]

      val data = rdd.mapPartitions { iter =>
        val text = new Text()
        iter.map { x =>
          text.set(x.toString)
          (NullWritable.get(), text)
        }
      }

      RDD.rddToPairRDDFunctions[NullWritable, Text](data)(nullWritableClassTag, textClassTag, null)
        .saveAsHadoopFile(
          path = path,
          keyClass = nullWritableClassTag.runtimeClass,
          valueClass = textClassTag.runtimeClass,
          outputFormatClass = outputFormatClassTag.runtimeClass.asInstanceOf[Class[_ <: OutputFormat[_, _]]],
          conf = jobConf,
          codec = codec
        )
    }

  }

}
