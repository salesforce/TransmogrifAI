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
import com.salesforce.op.features.types.{FeatureType, FeatureTypeSparkConverter}
import com.salesforce.op.features.{FeatureLike, FeatureSparkTypes, OPFeature}
import com.salesforce.op.readers.DataFrameFieldNames._
import com.twitter.algebird.Monoid
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataType, LongType, StructField, StructType}
import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession}
import org.joda.time.Duration
import org.slf4j.LoggerFactory

import scala.reflect.runtime.universe.WeakTypeTag


/**
 * Time column for aggregation
 *
 * @param name column name
 * @param keep should keep the column in result
 */
case class TimeColumn(name: String, keep: Boolean) {
  def this(feature: OPFeature, keep: Boolean) = this(feature.name, keep)

  def this(feature: OPFeature) = this(feature.name, keep = true)

  def this(name: String) = this(name, keep = true)
}

/**
 * Time based filter for conditional aggregation
 *
 * @param condition  condition time column
 * @param primary    primary time column
 * @param timeWindow time window for conditional aggregation
 */
case class TimeBasedFilter
(
  condition: TimeColumn,
  primary: TimeColumn,
  timeWindow: Duration
)

/**
 * Join Keys to use
 *
 * @param leftKey   key to use from left table
 * @param rightKey  key to use from right table (will always be the aggregation key
 * @param resultKey key of joined result
 */
case class JoinKeys
(
  leftKey: String = KeyFieldName,
  rightKey: String = KeyFieldName,
  resultKey: String = CombinedKeyName
) {

  /**
   * Is joining tables with parent child relations (left - parent, right - child)
   */
  def isParentChildJoin: Boolean = resultKey == KeyFieldName && leftKey == KeyFieldName && rightKey != KeyFieldName

  /**
   * Is joining tables with parent child relations (left - child, right - parent)
   */
  def isChildParentJoin: Boolean = resultKey == KeyFieldName && leftKey != KeyFieldName && rightKey == KeyFieldName

  /**
   * Is joining different tables containing different information on the same object
   */
  def isCombinedJoin: Boolean = resultKey == CombinedKeyName && leftKey == KeyFieldName && rightKey == KeyFieldName

  override def toString: String =
    s"${this.getClass.getSimpleName}(leftKey=$leftKey,rightKey=$rightKey,resultKey=$resultKey)"
}

/**
 * Join data reader trait
 *
 * @param leftReader  reader from left side of join (can also be join reader)
 * @param rightReader reader from right side of join (should be either conditional or aggregate reader)
 * @param joinKeys    join keys to use
 * @param joinType    type of join to perform
 * @tparam T Type of data read by left data reader
 * @tparam U Type of data read by right data reader
 */
private[op] abstract class JoinedReader[T, U]
(
  val leftReader: Reader[T],
  val rightReader: DataReader[U],
  val joinKeys: JoinKeys,
  val joinType: JoinType
)(implicit val wtt: WeakTypeTag[T], val wttu: WeakTypeTag[U]) extends Reader[T] {

  @transient protected lazy val log = LoggerFactory.getLogger(this.getClass)

  final def subReaders: Seq[DataReader[_]] = {
    val allReaders = Seq(leftReader.subReaders, rightReader.subReaders).flatten
    require(allReaders.size == allReaders.distinct.size, "Cannot have duplicate readers in joins")
    allReaders
  }

  protected val combineKeysUDF = udf { (k1: String, k2: String) => if (k1 == null) k2 else k1 }

  /**
   * Generate the dataframe that will be used in the OpPipeline calling this method
   *
   * @param rawFeatures features to generate from the dataset read in by this reader
   * @param opParams    op parameters
   * @param spark       spark instance to do the reading and conversion from RDD to Dataframe
   * @return A dataframe containing columns with all of the raw input features expected by the pipeline;
   *         a set of right join columns
   */
  protected def getJoinedData(
    rawFeatures: Array[OPFeature],
    opParams: OpParams
  )(implicit spark: SparkSession): (DataFrame, Array[String]) = {

    def getData(r: DataReader[_]): DataFrame = {
      val readerFeatures = rawFeatures.filter { f => getGenStage(f).tti.tpe.toString == r.fullTypeName }
      r.generateDataFrame(readerFeatures, opParams)
    }

    val (leftData, _) = leftReader match {
      case r: DataReader[_] => (getData(r), Array.empty[String])
      case r: JoinedReader[_, _] => r.getJoinedData(rawFeatures, opParams)
      case _ =>
        throw new RuntimeException(
          s"The reader type ${leftReader.getClass.getName} is not supported as leftReader for joins!")
    }

    val rightData = getData(rightReader).withColumnRenamed(KeyFieldName, RightKeyName)
    val rightCols = rightData.columns.filter(n => n != joinKeys.rightKey && n != RightKeyName)

    val joinedData = {
      val rightKey = if (joinKeys.rightKey == KeyFieldName) RightKeyName else joinKeys.rightKey
      leftData.join(
        rightData,
        leftData(joinKeys.leftKey) === rightData(rightKey),
        joinType.sparkJoinName
      )
    }
    val resultData =
      if (joinKeys.isParentChildJoin) joinedData.drop(RightKeyName, joinKeys.rightKey)
      else if (joinKeys.isChildParentJoin) joinedData.drop(RightKeyName)
      else if (joinKeys.isCombinedJoin) {
        joinedData
          .withColumn(joinKeys.resultKey, combineKeysUDF(col(joinKeys.leftKey), col(RightKeyName)))
          .drop(joinKeys.leftKey, RightKeyName)
          .withColumnRenamed(joinKeys.resultKey, joinKeys.leftKey)
      } else {
        throw new RuntimeException(s"Invalid key combination: $joinKeys")
      }
    resultData -> rightCols
  }

  /**
   * Generate the dataframe that will be used in the OpPipeline calling this method
   *
   * @param rawFeatures features to generate from the dataset read in by this reader
   * @param opParams    op parameters
   * @param spark       spark instance to do the reading and conversion from RDD to Dataframe
   * @return A dataframe containing columns with all of the raw input features expected by the pipeline
   */
  override def generateDataFrame(
    rawFeatures: Array[OPFeature],
    opParams: OpParams = new OpParams()
  )(implicit spark: SparkSession): DataFrame = {
    log.debug("Generating dataframe:\n Join type: {}\n Join keys: {}\n Raw features: {}",
      joinType, joinKeys, rawFeatures.map(_.name).mkString(","))
    val (joinedData, _) = getJoinedData(rawFeatures, opParams)
    joinedData
  }
}

/**
 * Holder class that contains individual data readers used for joins
 *
 * @param leftReader  reader from left side of join
 * @param rightReader reader from right side of join
 * @param joinKeys    join keys to use
 * @param joinType    type of join to perform
 * @tparam T Type of data read by left data reader
 * @tparam U Type of data read by right data reader
 */
private[op] class JoinedDataReader[T, U]
(
  leftReader: Reader[T],
  rightReader: DataReader[U],
  joinKeys: JoinKeys,
  joinType: JoinType
) extends JoinedReader[T, U](
  leftReader = leftReader, rightReader = rightReader, joinKeys = joinKeys, joinType = joinType
) {

  /**
   * Produces a new reader that will aggregate after joining the data
   *
   * @param timeFilter time filter for aggregation
   * @return A reader which will perform aggregation after loading the data
   */
  def withSecondaryAggregation(timeFilter: TimeBasedFilter): JoinedAggregateDataReader[T, U] = {
    new JoinedAggregateDataReader[T, U](
      leftReader = leftReader, rightReader = rightReader, joinKeys = joinKeys, joinType = joinType, timeFilter)
  }
}

/**
 * Holder class that contains individual data readers used for joins
 *
 * @param leftReader  reader from left side of join
 * @param rightReader reader from right side of join
 * @param joinKeys    join keys to use
 * @param joinType    type of join to perform
 * @param timeFilter  time based filter
 * @tparam T Type of data read by left data reader
 * @tparam U Type of data read by right data reader
 */
private[op] class JoinedAggregateDataReader[T, U]
(
  leftReader: Reader[T],
  rightReader: DataReader[U],
  joinKeys: JoinKeys,
  joinType: JoinType,
  val timeFilter: TimeBasedFilter
) extends JoinedReader[T, U](
  leftReader = leftReader, rightReader = rightReader, joinKeys = joinKeys, joinType = joinType
) {

  override def getJoinedData(
    rawFeatures: Array[OPFeature],
    opParams: OpParams
  )(implicit spark: SparkSession): (DataFrame, Array[String]) = {
    val (joined, rightCols) = super.getJoinedData(rawFeatures, opParams)
    val leftCols = (
      rawFeatures.map(_.name).toSet -- rightCols -- Set(joinKeys.leftKey, joinKeys.rightKey, joinKeys.resultKey)
      ).toArray
    log.debug("leftCols = {}, rightCols = {}", leftCols.mkString(","), rightCols.mkString(","): Any)
    postJoinAggregate(joined, rawFeatures, leftCols, rightCols) -> rightCols
  }

  protected def postJoinAggregate
  (
    joinedData: DataFrame,
    rawFeatures: Array[OPFeature],
    leftCols: Array[String],
    rightCols: Array[String]
  ): DataFrame = {
    val leftFeatures = rawFeatures.filter(f => leftCols.contains(f.name))
    val rightFeatures = rawFeatures.filter(f => rightCols.contains(f.name))

    val leftAggregators =
      if (joinKeys.isCombinedJoin) getConditionalAggregators(joinedData, leftFeatures, timeFilter)
      else {
        // generate dummy aggregators for parent data that keeps one copy of data for each key
        log.debug("Going to generate some dummy aggregators for left features: {}",
          leftFeatures.map(_.name).mkString(","))
        getAggregators(joinedData, leftFeatures, dummyAggregators = true)
      }
    // generate aggregators for child data
    val rightAggregators = getConditionalAggregators(joinedData, rightFeatures, timeFilter)
    val aggregators = leftAggregators ++ rightAggregators
    val featureNames = leftFeatures.map(_.name) ++ rightFeatures.map(_.name)
    val result =
      joinedData.groupBy(KeyFieldName)
        .agg(aggregators.head, aggregators.tail: _*)
        .toDF(KeyFieldName +: featureNames: _*)

    // drop un-wanted timestamp fields
    val timeFieldsToDrop = Seq(timeFilter.condition, timeFilter.primary).collect { case t if !t.keep => t.name }

    if (timeFieldsToDrop.isEmpty) result else result.drop(timeFieldsToDrop: _*)
  }

  protected def getAggregators(
    data: DataFrame, rawFeatures: Array[OPFeature], dummyAggregators: Boolean
  ): Seq[Column] = {
    rawFeatures.map { f =>
      val genStage = getGenStage(f)
      val monoid = genStage.aggregator.monoid
      val aggregator =
        if (dummyAggregators) {
          new DummyJoinedAggregator[FeatureType](
            feature = f.asInstanceOf[FeatureLike[FeatureType]],
            monoid = monoid.asInstanceOf[Monoid[FeatureType#Value]]
          )
        } else {
          new JoinedAggregator[FeatureType](
            feature = f.asInstanceOf[FeatureLike[FeatureType]],
            monoid = monoid.asInstanceOf[Monoid[FeatureType#Value]]
          )
        }
      aggregator(data(f.name))
    }.toSeq
  }

  protected def getConditionalAggregators(
    data: DataFrame, rawFeatures: Array[OPFeature], timeFilter: TimeBasedFilter
  ): Seq[Column] = {
    rawFeatures.map { f =>
      val genStage = getGenStage(f)
      val timeWindow = genStage.aggregateWindow.getOrElse(timeFilter.timeWindow)
      val monoid = genStage.aggregator.monoid
      val aggregator =
        new JoinedConditionalAggregator[FeatureType](
          feature = f.asInstanceOf[FeatureLike[FeatureType]],
          monoid = monoid.asInstanceOf[Monoid[FeatureType#Value]],
          timeWindow = timeWindow.getMillis
        )
      aggregator(data(f.name), data(timeFilter.primary.name), data(timeFilter.condition.name))
    }.toSeq
  }

}


// TODO: UserDefinedAggregateFunction is now deprecated in favor of Aggregator,
//  but that operates on Rows, not Columns. How would we redo this?
/**
 * Aggregator base for dataframe to use in JoinedAggregateDataReader
 *
 * @param feature feature to aggregate
 * @param monoid  the monoid attached to the aggregation phase of the feature to aggregate
 * @tparam O type of feature to aggregate
 */
private[op] abstract class JoinedAggregatorBase[O <: FeatureType]
(
  feature: FeatureLike[O], val monoid: Monoid[O#Value]
) extends UserDefinedAggregateFunction {
  protected val converter = FeatureTypeSparkConverter[O]()(feature.wtt)
  protected val initValue = converter.toSpark(converter.ftFactory.newInstance(monoid.zero))
  val inputSchema: StructType = FeatureSparkTypes.toStructType(feature)
  val bufferSchema: StructType = FeatureSparkTypes.toStructType(feature)
  val dataType: DataType = FeatureSparkTypes.sparkTypeOf(feature.wtt)
  protected def convertTypesMerge(v1: Any, v2: Any): Any
  override def deterministic: Boolean = true
  override def initialize(buffer: MutableAggregationBuffer): Unit = buffer(0) = initValue
  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    buffer(0) = convertTypesMerge(buffer.get(0), input.get(0))
  }
  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    buffer1(0) = convertTypesMerge(buffer1.get(0), buffer2.get(0))
  }
  override def evaluate(buffer: Row): Any = buffer.get(0)
}

/**
 * Aggregator for dataframe to use in [[JoinedAggregateDataReader]]
 *
 * @param feature feature to aggregate
 * @param monoid  the monoid attached to the aggregation phase of the feature to aggregate
 * @tparam O type of feature to aggregate
 */
private[op] class JoinedAggregator[O <: FeatureType]
(
  feature: FeatureLike[O], monoid: Monoid[O#Value]
) extends JoinedAggregatorBase[O](feature, monoid) {
  override protected def convertTypesMerge(v1: Any, v2: Any): Any = {
    val typedV1: O = converter.fromSpark(v1)
    val typedV2: O = converter.fromSpark(v2)
    val merged = monoid.plus(typedV1.value, typedV2.value)
    val mergedFeature: O = converter.ftFactory.newInstance(merged)
    converter.toSpark(mergedFeature)
  }
}

/**
 * Dummy aggregator for dataframe to use in [[JoinedAggregateDataReader]]
 *
 * @param feature feature to aggregate
 * @param monoid  the monoid attached to the aggregation phase of the feature to aggregate
 * @tparam O type of feature to aggregate
 */
private[op] class DummyJoinedAggregator[O <: FeatureType]
(
  feature: FeatureLike[O], monoid: Monoid[O#Value]
) extends JoinedAggregatorBase[O](feature, monoid) {
  override protected def convertTypesMerge(v1: Any, v2: Any): Any = v2
}

/**
 * Conditional aggregator for dataframe to use in [[JoinedAggregateDataReader]]
 *
 * @param feature feature to aggregate
 * @param monoid  the monoid attached to the aggregation phase of the feature to aggregate
 * @tparam O type of feature to aggregate
 */
private[op] class JoinedConditionalAggregator[O <: FeatureType]
(
  feature: FeatureLike[O], monoid: Monoid[O#Value], val timeWindow: Long
) extends JoinedAggregator[O](feature, monoid) {
  override val inputSchema: StructType = StructType(Array(
    FeatureSparkTypes.toStructField(feature),
    StructField("time", LongType),
    StructField("condition", LongType)
  ))
  val isResponse = feature.isResponse

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    val timeStamp = Option(input.getAs[Long](1)).getOrElse(0L) // time column
    val cutOff = Option(input.getAs[Long](2)).getOrElse(0L) // condition column
    buffer(0) = {
      if ((!isResponse && timeStamp < cutOff && timeStamp > cutOff - timeWindow) ||
        (isResponse && timeStamp >= cutOff && timeStamp < cutOff + timeWindow)) {
        convertTypesMerge(buffer.get(0), input.get(0))
      } else {
        buffer.get(0)
      }
    }
  }
}

