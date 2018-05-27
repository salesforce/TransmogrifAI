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

package com.salesforce.op.utils.io.avro

import java.net.URI

import com.salesforce.op.utils.io.DirectOutputCommitter
import com.salesforce.op.utils.spark.RichRDD._
import org.apache.avro.Schema
import org.apache.avro.generic.GenericRecord
import org.apache.avro.mapred._
import org.apache.avro.specific.SpecificData
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.NullWritable
import org.apache.hadoop.mapred.JobConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.reflect.ClassTag


/**
 * Contains methods for reading and writing avro records
 */
object AvroInOut {

  /**
   * This method reads avro records stored in directory/tablename/date into an RDD of generic/specific records
   * and then repartition to create copy rather than explicit deep copy
   *
   * @param path              Full path to where the avro records are stored
   * @param withCount         Whether or not to use a counter accumulator
   * @param maxPartitionCount max number of partitions
   * @param deepCopy          Whether or not to deep copy the avro records
   *                          instead of repartitioning the RDD (SPARK-1018)
   * @param persist           Whether or not to persist the rdd
   * @param sc                Spark session
   * @param ct                data class tag
   * @tparam T Generic/specific record type that records that should read as.
   * @return
   */
  def read[T <: GenericRecord](
    path: String,
    withCount: Boolean = false,
    maxPartitionCount: Int = 200,
    deepCopy: Boolean = false,
    persist: Boolean = true
  )(implicit sc: SparkSession, ct: ClassTag[T]): Option[RDD[T]] = {

    val fs = FileSystem.get(new java.net.URI(path), sc.sparkContext.hadoopConfiguration)
    val globStatus = fs.globStatus(new Path(path))

    if (Option(globStatus).exists(_.nonEmpty)) {
      Option(doRead(path = path, withCount = withCount,
        maxPartitionCount = maxPartitionCount, deepCopy = deepCopy, persist = persist))
    } else None
  }

  private[avro] def selectExistingPaths(path: String)(implicit sc: SparkSession): String = {
    val paths = path.split(",")
    val firstPath = paths.head.split("/")
    val bucket = firstPath.slice(0, math.min(4, firstPath.length)).mkString("/")
    val fs = try {
      FileSystem.get(new URI(bucket), sc.sparkContext.hadoopConfiguration)
    } catch {
      case ex: Exception => throw new IllegalArgumentException(s"Bad path $firstPath: ${ex.getMessage}")
    }
    val found = paths.filter(p => fs.exists(new Path(p)))
    if (found.isEmpty) throw new IllegalArgumentException("No valid directory found in the list of paths <<$path>>")
    found.mkString(",")
  }

  /**
   * This method reads avro records stored in directory/tablename/date into an RDD of specific records.
   * Allows user to pass in path of form "place1,place2,place3"
   * and then repartition to create copy rather than explicit deep copy
   *
   * @param path              Full path to where the avro records are stored
   * @param withCount         Whether or not to use a counter accumulator
   * @param maxPartitionCount max number of partitions
   * @param deepCopy          Whether or not to deep copy the avro records
   *                          instead of repartitioning the RDD (SPARK-1018)
   * @param persist           Whether or not to persist the rdd
   * @param sc                Spark session
   * @param ct                data class tag
   * @tparam T Specific record type that records that should read as.
   * @return
   */
  def readPathSeq[T <: GenericRecord](
    path: String,
    withCount: Boolean = false,
    maxPartitionCount: Int = 200,
    deepCopy: Boolean = false,
    persist: Boolean = true
  )(implicit sc: SparkSession, ct: ClassTag[T]): RDD[T] = {
    doRead(path = selectExistingPaths(path), withCount = withCount,
      maxPartitionCount = maxPartitionCount, deepCopy = deepCopy, persist = persist)
  }

  private def doRead[T <: GenericRecord](
    path: String,
    withCount: Boolean,
    maxPartitionCount: Int,
    deepCopy: Boolean,
    persist: Boolean)
    (implicit sc: SparkSession, ct: ClassTag[T]): RDD[T] = {
    def maybeCopy(r: T): T = if (deepCopy) SpecificData.get().deepCopy(r.getSchema, r) else r

    val records = sc.sparkContext.hadoopFile(path,
      classOf[AvroInputFormat[T]],
      classOf[AvroWrapper[T]],
      classOf[NullWritable]
    )

    val results =
      if (withCount) records.mapWithCount(r => maybeCopy(r._1.datum), "Number of records read")
      else records.map(r => maybeCopy(r._1.datum))

    // See Scaladoc on SparkSession.hadoopFile for information on Hadoop RecordReader's reuse of Writables
    // Spark's solution of a map transformation doesn't work on Avro records
    // So we either have to deep copy each record or repartition the RDD
    // https://issues.apache.org/jira/browse/SPARK-1018
    val res = if (!deepCopy) results.repartitionToMaxPartitions(maxPartitionCount) else results
    if (persist) res.persist() else res
  }

  implicit class AvroWriter[T <: GenericRecord](rdd: RDD[T]) {

    private def createJobConfFromContext(schema: String)(implicit sc: SparkSession) = {
      val jobConf = new JobConf(sc.sparkContext.hadoopConfiguration)
      jobConf.setOutputCommitter(classOf[DirectOutputCommitter])
      AvroJob.setOutputSchema(jobConf, new Schema.Parser().parse(schema))
      jobConf
    }

    /**
     * This method writes out RDDs of generic records as avro files to path.
     *
     * @param path Input directory where avro records should be written.
     * @param jobConf job config
     * @return
     */
    def writeAvro(path: String)(implicit jobConf: JobConf): Unit = {
      val avroData = rdd.map(ar => (new AvroKey(ar), NullWritable.get))
      avroData.saveAsHadoopFile(
        path,
        classOf[AvroWrapper[GenericRecord]],
        classOf[NullWritable],
        classOf[AvroOutputFormat[GenericRecord]],
        jobConf
      )
    }

    /**
     * This method writes out RDDs of generic records as avro files to path.
     *
     * @param path   Input directory where avro records should be written.
     * @param schema Avro schema string for records being written out.
     * @param sc     Spark Session
     * @return
     */
    def writeAvro(path: String, schema: String)(implicit sc: SparkSession): Unit = {
      val jobConf = createJobConfFromContext(schema)
      writeAvro(path)(jobConf)
    }

  }

}
