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

package com.salesforce.op

import com.salesforce.op.features.FeatureJsonHelper
import com.salesforce.op.filters.RawFeatureFilterResults
import com.salesforce.op.stages.{OpPipelineStageBase, OpPipelineStageWriter}
import enumeratum._
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.util.MLWriter
import org.json4s.JsonAST.{JArray, JObject, JString}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.json4s.{DefaultFormats, Formats}

/**
 * Writes the [[OpWorkflowModel]] to json format.
 * For now we will not serialize the parent of the model
 *
 * @note The features/stages must be sorted in topological order
 *
 * @param model workflow model to save
 */
class OpWorkflowModelWriter(val model: OpWorkflowModel) extends MLWriter {

  implicit val jsonFormats: Formats = DefaultFormats

  override protected def saveImpl(path: String): Unit = {
    sc.parallelize(Seq(toJsonString(path)), 1)
      .saveAsTextFile(OpWorkflowModelReadWriteShared.jsonPath(path))
  }

  /**
   * Json serialize model instance
   *
   * @param path to save the model and its stages
   * @return model json string
   */
  def toJsonString(path: String): String = pretty(render(toJson(path)))

  /**
   * Json serialize model instance
   *
   * @param path to save the model and its stages
   * @return model json
   */
  def toJson(path: String): JObject = {
    val FN = OpWorkflowModelReadWriteShared.FieldNames
    (FN.Uid.entryName -> model.uid) ~
      (FN.ResultFeaturesUids.entryName -> resultFeaturesJArray) ~
      (FN.BlacklistedFeaturesUids.entryName -> blacklistFeaturesJArray()) ~
      (FN.BlacklistedMapKeys.entryName -> blacklistMapKeys()) ~
      (FN.Stages.entryName -> stagesJArray(path)) ~
      (FN.AllFeatures.entryName -> allFeaturesJArray) ~
      (FN.Parameters.entryName -> model.getParameters().toJson(pretty = false)) ~
      (FN.TrainParameters.entryName -> model.trainingParams.toJson(pretty = false)) ~
      (FN.RawFeatureFilterResultsFieldName.entryName ->
        RawFeatureFilterResults.toJson(model.getRawFeatureFilterResults()))
  }

  private def resultFeaturesJArray(): JArray =
    JArray(model.getResultFeatures().map(_.uid).map(JString).toList)

  private def blacklistFeaturesJArray(): JArray =
    JArray(model.getBlacklist().map(_.uid).map(JString).toList)

  private def blacklistMapKeys(): JObject =
    JObject(model.getBlacklistMapKeys().map { case (k, vs) => k -> JArray(vs.map(JString).toList) }.toList)

  /**
   * Serialize all the workflow model stages
   *
   * @param path path to store the spark params for stages
   * @return array of serialized stages
   */
  private def stagesJArray(path: String): JArray = {
    val stages: Seq[OpPipelineStageBase] = model.getStages()
    val stagesJson: Seq[JObject] = stages
      .map(_.write.asInstanceOf[OpPipelineStageWriter].writeToJson(path))
      .filter(_.children.nonEmpty)
    JArray(stagesJson.toList)
  }

  /**
   * Gets all features to be serialized.
   *
   * @note Features should be topologically sorted
   * @return all features to be serialized
   */
  private def allFeaturesJArray: JArray =
    JArray(model.getAllFeatures().map(FeatureJsonHelper.toJson).toList)

}

/**
 * Shared functionality between [[OpWorkflowModelWriter]] and [[OpWorkflowModelReader]]
 */
private[op] object OpWorkflowModelReadWriteShared {
  def jsonPath(path: String): String = new Path(path, "op-model.json").toString

  /**
   * Model json field names
   */
  sealed abstract class FieldNames(override val entryName: String) extends EnumEntry

  /**
   * Model json field names
   */
  object FieldNames extends Enum[FieldNames] {
    val values = findValues
    case object Uid extends FieldNames("uid")
    case object ResultFeaturesUids extends FieldNames("resultFeaturesUids")
    case object BlacklistedFeaturesUids extends FieldNames("blacklistedFeaturesUids")
    case object BlacklistedMapKeys extends FieldNames("blacklistedMapKeys")
    case object Stages extends FieldNames("stages")
    case object AllFeatures extends FieldNames("allFeatures")
    case object Parameters extends FieldNames("parameters")
    case object TrainParameters extends FieldNames("trainParameters")
    case object RawFeatureFilterResultsFieldName extends FieldNames("rawFeatureFilterResults")
  }

}


/**
 * Writes the OpWorkflowModel into a specified path
 */
object OpWorkflowModelWriter {

  /**
   * Save [[OpWorkflowModel]] to path
   *
   * @param model     workflow model instance
   * @param path      path to save the model and its stages
   * @param overwrite should overwrite the destination
   */
  def save(model: OpWorkflowModel, path: String, overwrite: Boolean = true): Unit = {
    val w = new OpWorkflowModelWriter(model)
    val writer = if (overwrite) w.overwrite() else w
    writer.save(path)
  }

  /**
   * Serialize [[OpWorkflowModel]] to json
   *
   * @param model workflow model instance
   * @param path  path to save the model and its stages
   */
  def toJson(model: OpWorkflowModel, path: String): String = {
    new OpWorkflowModelWriter(model).toJsonString(path)
  }
}
