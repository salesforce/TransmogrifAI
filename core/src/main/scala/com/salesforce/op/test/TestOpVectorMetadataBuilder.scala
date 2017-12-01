/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.test

import com.salesforce.op.features.OPFeature
import com.salesforce.op.stages.OpPipelineStage
import com.salesforce.op.test.TestOpVectorColumnType._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}


/**
 * Represents a type of column associated with a feature in a vector metadata.
 */
sealed trait TestOpVectorColumnType

object TestOpVectorColumnType {

  /**
   * Represents just a plain column with no indicator (e.g. (featureName)_operationName)
   */
  case object RootCol extends TestOpVectorColumnType

  /**
   * Represents a column with an indicator (e.g. (featurename)_operationName_indicatorValue)
   *
   * @param name Name of the indicator value
   */
  case class IndCol(name: Option[String]) extends TestOpVectorColumnType

  /**
   * Represents a column with an value but no defined grouping beyond the parent feature name
   *
   * @param name Name of the indicator value
   */
  case class IndVal(name: Option[String]) extends TestOpVectorColumnType


  /**
   * Represents a column with an indicator (e.g. (featurename)_operationName_indicatorValue), but the
   * resulting [[OpVectorColumnMetadata]] should not contain the indicator value (this is done in
   * [[com.salesforce.op.stages.impl.feature.OPCollectionHashingVectorizer]], for instance, where there are indicators
   * but these indicators do not represent mutually exclusive values)
   *
   * @param name Name of the indicator value
   */
  case class PivotColNoInd(name: String) extends TestOpVectorColumnType

  /**
   * Represents a column with an indicator, but the resulting [[OpVectorColumnMetadata]] should have a different
   * group than the parent feature's name.
   *
   * @param name Name of the indicator
   * @param groupName Name of the indicator's group
   */
  case class IndColWithGroup(name: Option[String], groupName: String) extends TestOpVectorColumnType

}

/**
 * Helps construct [[OpVectorMetadata]] as expected from a stage
 */
object TestOpVectorMetadataBuilder {

  /**
   * Construct an [[OpVectorMetadata]] from the given stage and features, along with any columns associated with each
   * feature.
   *
   * @param stage The stage to construct from
   * @param fs A seq of tuples. The first element is the feature, and the second element is all the columns that
   *           the vectorizer should produce from that feature
   * @return OpVectorMetadata
   */
  def apply(stage: OpPipelineStage[_], fs: (OPFeature, List[TestOpVectorColumnType])*): OpVectorMetadata = {
    withOpNames(stage,
      fs.map { case (f, cols) => (f, stage.operationName, cols)
    }: _*)
  }

  /**
   * Construct an [[OpVectorMetadata]] from the given stage and features, along with any columns associated with each
   * feature. This lets the user provide the operation name for the column names, instead of it being assumed to be
   * the stage's operation name.
   *
   * @param stage The stage to construct from
   * @param fs A seq of tuples. The first element is the feature, the second element is the operation name that
   *           produced it, and the third element is all the columns that the vectorizer should produce from that
   *           feature
   * @return OpVectorMetadata
   */
  def withOpNames(
    stage: OpPipelineStage[_],
    fs: (OPFeature, String, List[TestOpVectorColumnType])*
  ): OpVectorMetadata = {
    val cols = for {
      (f, opName, colNames) <- fs.toArray
      col <- colNames
    } yield OpVectorColumnMetadata(
      parentFeatureName = Seq(f.name),
      parentFeatureType = Seq(f.typeName),
      indicatorGroup = col match {
        case RootCol => None
        case PivotColNoInd(_) => None
        case IndCol(name) => Option(f.name)
        case IndVal(_) => None
        case IndColWithGroup(_, groupName) => Option(groupName)
      },
      indicatorValue = col match {
        case RootCol => None
        case PivotColNoInd(_) => None
        case IndCol(maybeName) => maybeName
        case IndVal(name) => name
        case IndColWithGroup(maybeName, _) => maybeName
      }
    )
    val hist = fs.toArray.map(f => f._1.name -> f._1.history()).toMap
    OpVectorMetadata(stage.outputName, cols, hist)
  }


}
