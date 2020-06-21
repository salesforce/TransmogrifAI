/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.utils.io.avro

import java.net.URI

import com.salesforce.op.utils.spark.RichRDD._
import org.apache.avro.Schema
import org.apache.avro.generic.GenericRecord
import org.apache.avro.mapred.AvroKey
import org.apache.avro.mapreduce.{AvroJob, AvroKeyInputFormat, AvroKeyOutputFormat}
import org.apache.avro.specific.SpecificData
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.NullWritable
import org.apache.hadoop.mapreduce.Job
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.reflect.ClassTag
import scala.util.{Failure, Success, Try}


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
    val paths = path.split(',')
    val fsSeq = paths.map(path => Try(FileSystem.get(new URI(path), sc.sparkContext.hadoopConfiguration)))
    val validFS = fsSeq.filter(p => p.isSuccess)
    if (validFS.nonEmpty) {
      val fs = validFS.head.get // just get the first valid FS instance
      val validPaths: Array[String] = paths.filter(p => fs.exists(new Path(p)))
      if (validPaths.isEmpty) { // no path exists
        throw new IllegalArgumentException(s"No valid directory found in path '$path'")
      }
      validPaths.mkString(",") // found valid or empty paths
    }
    else { // no path from the comma separated path string argument was readable by FileSystem get
      // just get the first error message
      val fsFailed: Try[FileSystem] = fsSeq.head
      fsFailed match {
        case Failure(message) => throw new IllegalArgumentException(s"No readable path found : $message")
        case Success(_) => throw new IllegalArgumentException("This shouldn't happen since no readable paths were found earlier.")
      }
    }
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

    val records = sc.sparkContext.newAPIHadoopFile(path,
      classOf[AvroKeyInputFormat[T]],
      classOf[AvroKey[T]],
      classOf[NullWritable],
      sc.sparkContext.hadoopConfiguration
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

    private def writeAvro(path: String)(implicit job: Job): Unit = {
      val avroData = rdd.map(ar => (new AvroKey(ar), NullWritable.get))
      avroData.saveAsNewAPIHadoopFile(
        path,
        classOf[AvroKey[T]],
        classOf[NullWritable],
        classOf[AvroKeyOutputFormat[T]],
        job.getConfiguration
      )
    }

    /**
     * This method writes out RDDs of generic records as avro files to path.
     *
     * @param path   Input directory where avro records should be written.
     * @param schema Avro schema string for records being written out.
     * @return
     */
    def writeAvro(path: String, schema: String)
      (implicit job: Job = Job.getInstance(rdd.sparkContext.hadoopConfiguration)): Unit = {
      AvroJob.setOutputKeySchema(job, new Schema.Parser().parse(schema))
      writeAvro(path)(job)
    }

  }

}
