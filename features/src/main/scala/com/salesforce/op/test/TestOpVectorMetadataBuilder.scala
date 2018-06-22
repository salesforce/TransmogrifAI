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

package com.salesforce.op.test

import com.salesforce.op.FeatureHistory
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
    val hist = fs.map{ case (f, _, _) =>
      f.name -> f.history().merge(FeatureHistory(Seq.empty, Seq(stage.stageName)))
    }
    withOpNamesAndHist(stage, hist.toMap, fs: _*)
  }

  /**
   * Construct an [[OpVectorMetadata]] from the given stage and features, along with any columns associated with each
   * feature. This lets the user provide the operation name for the column names, instead of it being assumed to be
   * the stage's operation name.
   *
   * @param stage The stage to construct from
   * @param hist  The history of the parent features
   * @param fs A seq of tuples. The first element is the feature, the second element is the operation name that
   *           produced it, and the third element is all the columns that the vectorizer should produce from that
   *           feature
   * @return OpVectorMetadata
   */
  def withOpNamesAndHist(
    stage: OpPipelineStage[_],
    hist: Map[String, FeatureHistory],
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
    OpVectorMetadata(stage.getOutputFeatureName, cols, hist)
  }

}
