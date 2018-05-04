/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op

import com.salesforce.op.OpWorkflowModelReadWriteShared.FieldNames._
import com.salesforce.op.features.{FeatureJsonHelper, OPFeature, TransientFeature}
import com.salesforce.op.stages.{OpPipelineStageReader, _}
import OpPipelineStageReadWriteShared._
import org.apache.spark.ml.util.MLReader
import org.json4s.JsonAST.{JArray, JNothing, JValue}
import org.json4s.jackson.JsonMethods.parse

import scala.util.{Failure, Success, Try}

/**
 * Reads OpWorkflowModelWriter serialized OpWorkflowModel objects by path and JValue.
 * This will only work if the features were serialized in topological order.
 * NOTE: The FeatureGeneratorStages will not be recovered into the Model object, because they are part of each feature.
 *
 * @param workflow the workflow that produced the trained model
 */
class OpWorkflowModelReader(val workflow: OpWorkflow) extends MLReader[OpWorkflowModel] {

  /**
   * Load a previously trained workflow model from path
   *
   * @param path to the trained workflow model
   * @return workflow model
   */
  final override def load(path: String): OpWorkflowModel = {
    Try(sc.textFile(OpWorkflowModelReadWriteShared.jsonPath(path), 1).collect().mkString).flatMap(loadJson) match {
      case Failure(error) => throw new RuntimeException(s"Failed to load Workflow: ${error.getMessage}", error)
      case Success(wf) => wf
    }
  }

  /**
   * Load a previously trained workflow model from json
   *
   * @param json json of the trained workflow model
   * @return workflow model
   */
  def loadJson(json: String): Try[OpWorkflowModel] = Try(parse(json)).flatMap(loadJson)

  /**
   * Load Workflow instance from json
   *
   * @param json json value
   * @return workflow model instance
   */
  def loadJson(json: JValue): Try[OpWorkflowModel] = {
    for {
      trainParams <- OpParams.fromString((json \ TrainParameters.entryName).extract[String])
      params <- OpParams.fromString((json \ Parameters.entryName).extract[String])
      model <- Try(new OpWorkflowModel(uid = (json \ Uid.entryName).extract[String], trainParams))
      (stages, resultFeatures) <- Try(resolveFeaturesAndStages(json))
      blacklist <- Try(resolveBlacklist(json))
    } yield model
      .setStages(stages.filterNot(_.isInstanceOf[FeatureGeneratorStage[_, _]]))
      .setFeatures(resultFeatures)
      .setParameters(params)
      .setBlacklist(blacklist)
  }

  private def resolveBlacklist(json: JValue): Array[OPFeature] = {
    if ((json \ BlacklistedFeaturesUids.entryName) != JNothing) { // for backwards compatibility
      val blacklistIds = (json \ BlacklistedFeaturesUids.entryName).extract[JArray].arr
      val allFeatures = workflow.rawFeatures ++ workflow.blacklistedFeatures ++
        workflow.stages.flatMap(s => s.getInputFeatures()) ++
        workflow.resultFeatures
      blacklistIds.flatMap(uid => allFeatures.find(_.uid == uid.extract[String])).toArray
    } else {
      Array.empty[OPFeature]
    }
  }

  private def resolveFeaturesAndStages(json: JValue): (Array[OPStage], Array[OPFeature]) = {
    val stages = loadStages(json)
    val stagesMap = stages.map(stage => stage.uid -> stage).toMap[String, OPStage]
    val featuresMap = resolveFeatures(json, stagesMap)
    resolveStages(stages, featuresMap)

    val resultIds = (json \ ResultFeaturesUids.entryName).extract[Array[String]]
    val resultFeatures = featuresMap.filterKeys(resultIds.toSet).values

    stages.toArray -> resultFeatures.toArray
  }

  private def loadStages(json: JValue): Seq[OPStage] = {
    val stagesJs = (json \ Stages.entryName).extract[JArray].arr
    val recoveredStages = stagesJs.map(j => {
      val stageUid = (j \ FieldNames.Uid.entryName).extract[String]
      val originalStage = workflow.stages.find(_.uid == stageUid)
      originalStage match {
        case Some(os) => new OpPipelineStageReader(os).loadFromJson(j).asInstanceOf[OPStage]
        case None => throw new RuntimeException(s"Workflow does not contain a stage with uid: $stageUid")
      }
    })
    val generators = workflow.rawFeatures.map(_.originStage)
    generators ++ recoveredStages
  }

  private def resolveFeatures(json: JValue, stages: Map[String, OPStage]): Map[String, OPFeature] = {
    val results = (json \ AllFeatures.entryName).extract[JArray].arr
    // should have been serialized in topological order
    // so that parent features can be used to construct each new feature
    results.foldLeft(Map.empty[String, OPFeature])((featMap, feat) =>
      FeatureJsonHelper.fromJson(feat, stages, featMap) match {
        case Success(f) => featMap + (f.uid -> f)
        case Failure(e) => throw new RuntimeException(s"Error resolving feature: $feat", e)
      }
    )
  }

  private def resolveStages(stages: Seq[OPStage], featuresMap: Map[String, OPFeature]): Unit = {
    for {stage <- stages} {
      val inputIds = stage.getTransientFeatures().map(_.uid)
      val inFeatures = inputIds.map(id => TransientFeature(featuresMap(id))) // features are order dependent
      stage.set(stage.inputFeatures, inFeatures)
    }
  }


}
