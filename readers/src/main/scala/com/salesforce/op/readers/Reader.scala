/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.readers

import com.salesforce.op.OpParams
import com.salesforce.op.features.OPFeature
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.FeatureGeneratorStage
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.reflect.runtime.universe.WeakTypeTag


trait Reader[T] extends Serializable {

  implicit val wtt: WeakTypeTag[T]

  /**
   * All the reader's sub readers (used in joins)
   * @return sub readers
   */
  def subReaders: Seq[DataReader[_]]

  /**
   * Outer join
   *
   * @param other    reader from right side of join
   * @param joinKeys join keys to use
   * @tparam U Type of data read by right data reader
   * @return joined reader
   */
  final def outerJoin[U](other: DataReader[U], joinKeys: JoinKeys = JoinKeys()): JoinedDataReader[T, U] =
    join(other, joinType = JoinTypes.Outer, joinKeys)

  /**
   * Left Outer join
   *
   * @param other    reader from right side of join
   * @param joinKeys join keys to use
   * @tparam U Type of data read by right data reader
   * @return joined reader
   */
  final def leftOuterJoin[U](other: DataReader[U], joinKeys: JoinKeys = JoinKeys()): JoinedDataReader[T, U] =
    join(other, joinType = JoinTypes.LeftOuter, joinKeys)

  /**
   * Inner join
   *
   * @param other    reader from right side of join
   * @param joinKeys join keys to use
   * @tparam U Type of data read by right data reader
   * @return joined reader
   */
  final def innerJoin[U](other: DataReader[U], joinKeys: JoinKeys = JoinKeys()): JoinedDataReader[T, U] =
    join(other, joinType = JoinTypes.Inner, joinKeys)

  /**
   * Join readers
   *
   * @param other    reader from right side of join
   * @param joinKeys join keys to use
   * @param joinType type of join to perform
   * @tparam U Type of data read by right data reader
   * @return joined reader
   */
  final protected def join[U](
    other: DataReader[U],
    joinType: JoinType,
    joinKeys: JoinKeys = JoinKeys()
  ): JoinedDataReader[T, U] = {
    val joinedReader =
      new JoinedDataReader[T, U](leftReader = this, rightReader = other, joinKeys = joinKeys, joinType = joinType)
    assert(joinedReader.leftReader.subReaders
      .forall(r => r.fullTypeName != joinedReader.rightReader.fullTypeName),
      "All joins must be for readers of different objects - self joins are not supported"
    )
    joinedReader
  }

  /**
   * Generate the dataframe that will be used in the OpPipeline calling this method
   *
   * @param rawFeatures features to generate from the dataset read in by this reader
   * @param opParams    op parameters
   * @param spark       spark instance to do the reading and conversion from RDD to Dataframe
   * @return A dataframe containing columns with all of the raw input features expected by the pipeline
   */
  def generateDataFrame(rawFeatures: Array[OPFeature], opParams: OpParams)(implicit spark: SparkSession): DataFrame

  protected[op] def getGenStage[I](f: OPFeature): FeatureGeneratorStage[I, _ <: FeatureType] = {
    f.originStage match {
      case fg: FeatureGeneratorStage[_, _] => fg.asInstanceOf[FeatureGeneratorStage[I, _ <: FeatureType]]
      case _ => throw new IllegalArgumentException(
        s"All raw features must have origin stage of type ${
          classOf[FeatureGeneratorStage[I, _ <: FeatureType]].getSimpleName
        }"
      )
    }
  }
}
