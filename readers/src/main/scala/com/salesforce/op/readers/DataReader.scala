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

package com.salesforce.op.readers

import com.salesforce.op.OpParams
import com.salesforce.op.aggregators.CutOffTime
import com.salesforce.op.features.types.FeatureTypeSparkConverter
import com.salesforce.op.features.{FeatureSparkTypes, OPFeature}
import com.salesforce.op.readers.DataFrameFieldNames._
import com.salesforce.op.utils.date.DateTimeUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.util.ClosureUtils
import org.joda.time.{DateTimeConstants, Duration}

import scala.util.Failure


/**
 * DataReaders must specify:
 * 1. An optional path to read from
 * 2. A function for extracting the key from the records being read
 * 3. The read method to be used for reading the data
 *
 * @tparam T
 */
trait DataReader[T] extends Reader[T] with ReaderKey[T] {

  /**
   * Default optional read path
   * @return optional read path
   */
  def readPath: Option[String]

  /**
   * Function which reads raw data from specified location to use in Dataframe creation, i.e. [[generateDataFrame]] fun.
   * This function returns either RDD or Dataset of the type specified by this reader.
   * It can be overwritten to carry out any special logic required for the reader
   * (ie filters needed to produce the specified reader type).
   *
   * @param params parameters used to carry out specialized logic in reader (passed in from workflow)
   * @param spark  spark instance to do the reading and conversion from RDD to Dataframe
   * @return either RDD or Dataset of type T
   */
  protected def read(params: OpParams)(implicit spark: SparkSession): Either[RDD[T], Dataset[T]]

  /**
   * Function which reads raw data from specified location to use in Dataframe creation, i.e. [[generateDataFrame]] fun.
   * This function returns a RDD of the type specified by this reader.
   *
   * @param params parameters used to carry out specialized logic in reader (passed in from workflow)
   * @param sc     spark session
   * @return RDD of type T
   */
  final def readRDD(params: OpParams = new OpParams())(implicit sc: SparkSession): RDD[T] = {
    read(params) match {
      case Left(rdd) => rdd
      case Right(ds) => ds.rdd
    }
  }

  /**
   * Function which reads raw data from specified location to use in Dataframe creation, i.e. [[generateDataFrame]] fun.
   * This function returns a Dataset of the type specified by this reader.
   *
   * @param params parameters used to carry out specialized logic in reader (passed in from workflow)
   * @param sc     spark session
   * @return Dataset of type T
   */
  final def readDataset(params: OpParams = new OpParams())
    (implicit sc: SparkSession, encoder: Encoder[T]): Dataset[T] = {
    read(params) match {
      case Left(rdd) => sc.createDataset(rdd)
      case Right(ds) => ds
    }
  }

  /**
   * Derives DataFrame schema for raw features.
   *
   * @param rawFeatures feature array representing raw feature-data
   * @return a StructType instance
   */
  final protected def getSchema(rawFeatures: Array[OPFeature]): StructType = {
    val keyField = StructField(name = KeyFieldName, dataType = StringType, nullable = false)
    val featureFields = rawFeatures.map(FeatureSparkTypes.toStructField(_))
    StructType(keyField +: featureFields)
  }

  /**
   * Default method for extracting the path used in read method. The path is taken in the following order
   * of priority: readerPath, params
   *
   * @param params
   * @return final path to use
   */
  final protected def getFinalReadPath(params: OpParams): String = {
    val finalPath = readPath.orElse(getReaderParams(params).flatMap(_.path))
    require(finalPath.isDefined, "The path is not set")
    finalPath.get
  }

  /**
   * Function to repartition the data based on the op params of this reader
   *
   * @param data   rdd
   * @param params op params
   * @return maybe repartitioned rdd
   */
  final protected def maybeRepartition(data: RDD[T], params: OpParams): RDD[T] =
    (for {
      params <- getReaderParams(params)
      partitions <- params.partitions
    } yield data.repartition(partitions)).getOrElse(data)

  /**
   * Function to repartition the data based on the op params of this reader
   *
   * @param data   dataset
   * @param params op params
   * @return maybe repartitioned dataset
   */
  final protected def maybeRepartition(data: Dataset[T], params: OpParams): Dataset[T] =
    (for {
      params <- getReaderParams(params)
      partitions <- params.partitions
    } yield data.repartition(partitions)).getOrElse(data)

  /**
   * Generate the Dataframe that will be used in the OpPipeline calling this method
   *
   * @param rawFeatures features to generate from the dataset read in by this reader
   * @param opParams    op parameters
   * @param spark       spark instance to do the reading and conversion from RDD to Dataframe
   * @return A Dataframe containing columns with all of the raw input features expected by the pipeline
   */
  override def generateDataFrame(
    rawFeatures: Array[OPFeature],
    opParams: OpParams = new OpParams()
  )(implicit spark: SparkSession): DataFrame = {
    val rawData = read(opParams)
    val schema = getSchema(rawFeatures)

    rawData match {
      case Left(rdd) =>
        val d = rdd.flatMap(record => generateRow(key(record), record, rawFeatures, schema))
        spark.createDataFrame(d, schema)
      case Right(ds) =>
        val inputSchema = ds.schema.fields
        if (schema.forall(fn => inputSchema.exists( // check if features to be extracted already exist in dataframe
          fi => fn.name == fi.name && fn.dataType == fi.dataType && fn.nullable == fi.nullable)
        )) {
          val names = schema.fields.map(_.name).toSeq
          ds.select(names.head, names.tail: _*)
        } else {
          implicit val rowEnc = RowEncoder(schema)
          val df = ds.flatMap(record => generateRow(key(record), record, rawFeatures, schema))
          spark.createDataFrame(df.rdd, schema) // because the spark row encoder does not preserve metadata
        }
    }
  }

  protected def generateRow(key: String, record: T, rawFeatures: Array[OPFeature], schema: StructType): Option[Row] = {
    val vals = rawFeatures.map { f =>
      val featureGen = getGenStage[T](f)
      val extracted = featureGen.extractFn(record)
      FeatureTypeSparkConverter.toSpark(extracted)
    }
    Some(new GenericRowWithSchema(key +: vals, schema))
  }
}

/**
 * Readers that extend this can be used as right hand side arguments for joins and so
 * should do aggregation on the key to return only a single value
 *
 * @tparam T
 */
trait AggregatedReader[T] extends DataReader[T] {
  implicit val strEnc = Encoders.STRING
  implicit val seqEnc = Encoders.kryo[Seq[T]]
  implicit val tupEnc = Encoders.tuple[String, Seq[T]](strEnc, seqEnc)

  /**
   * Generate the Dataframe that will be used in the OpPipeline calling this method
   *
   * @param rawFeatures features to generate from the dataset read in by this reader
   * @param opParams    op parameters
   * @param spark       spark instance to do the reading and conversion from RDD to Dataframe
   * @return A Dataframe containing columns with all of the raw input features expected by the pipeline
   */
  final override def generateDataFrame(
    rawFeatures: Array[OPFeature],
    opParams: OpParams = new OpParams()
  )(implicit spark: SparkSession): DataFrame = {
    val rawData = read(opParams)
    val schema = getSchema(rawFeatures)

    rawData match {
      case Left(rdd) =>
        val rowRDD =
          rdd.map(record => (key(record), Seq(record)))
            .reduceByKey(_ ++ _)
            .flatMap { case (key, records) => generateRow(key, records, rawFeatures, schema) }
        spark.createDataFrame(rowRDD, schema)

      case Right(ds) =>
        implicit val rowEnc = RowEncoder(schema)
        ds.map(record => (key(record), Seq(record)))
          .groupByKey(_._1)
          .reduceGroups((l: (String, Seq[T]), r: (String, Seq[T])) => (l._1, l._2 ++ r._2))
          .flatMap { case (key, (_, records)) => generateRow(key, records, rawFeatures, schema) }
    }
  }

  protected def generateRow(
    key: String, records: Seq[T],
    rawFeatures: Array[OPFeature],
    schema: StructType
  ): Option[Row]

}

/**
 * DataReader to use for event type data, with multiple records per key
 *
 * @tparam T
 */
trait AggregateDataReader[T] extends AggregatedReader[T] {

  /**
   * Aggregate data reader params
   */
  def aggregateParams: AggregateParams[T]

  final override def generateRow(key: String, records: Seq[T], rawFeatures: Array[OPFeature],
    schema: StructType): Option[Row] = {
    val AggregateParams(timeStampFn, cutOffTime) = aggregateParams
    val vals: Array[Any] = rawFeatures.map { f =>
      val featureAgg = getGenStage[T](f).featureAggregator
      val extracted = featureAgg.extract(records = records, timeStampFn = timeStampFn, cutOffTime = cutOffTime)
      FeatureTypeSparkConverter.toSpark(extracted)
    }
    Some(new GenericRowWithSchema(key +: vals, schema))
  }
}

/**
 * Aggregate data reader params
 *
 * @param timeStampFn An additional timeStamp function for extracting the timestamp of the event
 * @param cutOffTime  A cut off time to be used for aggregating features extracted from the events
 *                    - Predictor variables will be aggregated from events up until the cut off time
 *                    - Response variables will be aggregated from events following the cut off time
 * @tparam T
 */
case class AggregateParams[T](timeStampFn: Option[T => Long], cutOffTime: CutOffTime)

/**
 * DataReader to use for event type data, when modeling conditional probabilities.
 * Predictor variables will be aggregated from events up until the occurrence of the condition.
 * Response variables will be aggregated from events following the occurrence of the condition.
 *
 * @tparam T
 */
trait ConditionalDataReader[T] extends AggregatedReader[T] {

  /**
   * Conditional data reader params
   */
  def conditionalParams: ConditionalParams[T]

  final override def generateRow(key: String, records: Seq[T],
    rawFeatures: Array[OPFeature], schema: StructType): Option[Row] = {
    val ConditionalParams(
    timeStampFn, targetCondition, responseWindow,
    predictorWindow, timeStampToKeep, cutOffTimeFn,
    dropIfTargetConditionNotMet) = conditionalParams

    val rawTargetTimes = records.collect { case record if targetCondition(record) => timeStampFn(record) }

    if (rawTargetTimes.isEmpty && dropIfTargetConditionNotMet) None
    else {
      val cutOff: CutOffTime = cutOffTimeFn.map(_(key, records)).getOrElse(cutOffTime(rawTargetTimes, timeStampToKeep))

      val featureVals = rawFeatures.map { f =>
        val featureAgg = getGenStage[T](f).featureAggregator
        val extracted = featureAgg.extract(
          records = records,
          timeStampFn = Some(timeStampFn),
          cutOffTime = cutOff,
          responseWindow = responseWindow,
          predictorWindow = predictorWindow
        )
        FeatureTypeSparkConverter.toSpark(extracted)
      }
      Some(new GenericRowWithSchema(key +: featureVals, schema))
    }
  }

  private def cutOffTime(rawTargetTimes: Seq[Long], timeStampToKeep: TimeStampToKeep): CutOffTime = {
    import TimeStampToKeep._
    val targetTime: Long =
      if (rawTargetTimes.isEmpty) DateTimeUtils.now().getMillis
      else timeStampToKeep match {
        case Min => rawTargetTimes.min
        case Max => rawTargetTimes.max
        case Random => rawTargetTimes(scala.util.Random.nextInt(rawTargetTimes.size)) // TODO should this be seeded?
      }
    CutOffTime.UnixEpoch(targetTime)
  }
}

/**
 * Conditional data reader params
 *
 * @param timeStampFn                 function for extracting the timestamp from an event
 * @param targetCondition             function for identifying if the condition is met
 * @param responseWindow              optional size of time window over which the response variable is to be aggregated
 * @param predictorWindow             optional size of time window over which the predictor variables
 *                                    are to be aggregated
 * @param timeStampToKeep             if a particular key met the condition multiple times, which of the times
 *                                    would you like to use in the training set
 * @param cutOffTimeFn                optional function to compute the cutoff value based on key and aggregated
 *                                    sequence of events for that key
 * @param dropIfTargetConditionNotMet do not generate feature vectors for keys in training set
 *                                    where the target condition is not met. If set to false,
 *                                    and condition is not met, features for those
 */
case class ConditionalParams[T]
(
  timeStampFn: T => Long,
  targetCondition: T => Boolean,
  responseWindow: Option[Duration] = Some(Duration.standardDays(DateTimeConstants.DAYS_PER_WEEK)),
  predictorWindow: Option[Duration] = Some(Duration.standardDays(DateTimeConstants.DAYS_PER_WEEK)),
  timeStampToKeep: TimeStampToKeep = TimeStampToKeep.Random,
  cutOffTimeFn: Option[(String, Seq[T]) => CutOffTime] = None,
  dropIfTargetConditionNotMet: Boolean = false
) {
  // Validate function params
  Seq(timeStampFn, targetCondition, cutOffTimeFn.getOrElse(identity _)).foreach(function =>
    ClosureUtils.checkSerializable(function) match {
      case Failure(e) => throw new IllegalArgumentException("Function is not serializable", e)
      case _ =>
    }
  )
}
