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


import com.salesforce.op.OpWorkflowModelReadWriteShared.{FieldNames => FN}
import com.salesforce.op.OpWorkflowModelReadWriteShared.FieldNames._
import com.salesforce.op.OpWorkflowModelReadWriteShared.DeprecatedFieldNames._
import com.salesforce.op.features.{FeatureJsonHelper, OPFeature, TransientFeature}
import com.salesforce.op.filters.{FeatureDistribution, RawFeatureFilterResults}
import com.salesforce.op.stages.OpPipelineStageReaderWriter._
import com.salesforce.op.stages._
import org.apache.commons.io.IOUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.compress.CompressionCodecFactory
import org.json4s.JsonAST.{JArray, JNothing, JValue}
import org.json4s.jackson.JsonMethods.parse

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.{Failure, Success, Try}

/**
 * Reads OpWorkflowModelWriter serialized OpWorkflowModel objects by path and JValue.
 * This will only work if the features were serialized in topological order.
 * NOTE: The FeatureGeneratorStages will not be recovered into the Model object, because they are part of each feature.
 *
 * @param workflowOpt optional workflow that produced the trained model
 * @param asSpark if true will load as spark models if false will load as Mlleap stages for spark wrapped stages
 */
class OpWorkflowModelReader(val workflowOpt: Option[OpWorkflow], val asSpark: Boolean = true) {

  /**
   * Load a previously trained workflow model from path
   *
   * @param path to the trained workflow model
   * @return workflow model
   */
  final def load(path: String): OpWorkflowModel = {
    implicit val conf = new org.apache.hadoop.conf.Configuration()
    Try(FileReader.loadFile(OpWorkflowModelReadWriteShared.jsonPath(path)))
      .flatMap(loadJson(_, path = path)) match {
      case Failure(error) => throw new RuntimeException(s"Failed to load Workflow from path '$path'", error)
      case Success(wf) => wf

    }
  }

  /**
   * Load a previously trained workflow model from json
   *
   * @param json json of the trained workflow model
   * @param path to the trained workflow model
   * @return workflow model
   */
  def loadJson(json: String, path: String): Try[OpWorkflowModel] = {
    Try(parse(json)).flatMap(loadJson(_, path = path))
  }

  /**
   * Load Workflow instance from json
   *
   * @param json json value
   * @param path to the trained workflow model
   * @return workflow model instance
   */
  def loadJson(json: JValue, path: String): Try[OpWorkflowModel] = {
    for {
      trainingParams <- OpParams.fromString((json \ TrainParameters.entryName).extract[String])
      params <- OpParams.fromString((json \ Parameters.entryName).extract[String])
      model <- Try(new OpWorkflowModel(uid = (json \ Uid.entryName).extract[String], trainingParams))
      stages <- loadStages(json, workflowOpt, path)
      resolvedFeatures <- resolveFeatures(json, stages)
      resultFeatures <- resolveResultFeatures(json, resolvedFeatures)
      blocklist <- resolveBlocklist(json, workflowOpt, resolvedFeatures, path)
      blocklistMapKeys <- resolveBlocklistMapKeys(json)
      rffResults <- resolveRawFeatureFilterResults(json)
    } yield model
      .setStages(stages.filterNot(_.isInstanceOf[FeatureGeneratorStage[_, _]]))
      .setFeatures(resultFeatures)
      .setParameters(params)
      .setBlocklist(blocklist)
      .setBlocklistMapKeys(blocklistMapKeys)
      .setRawFeatureFilterResults(rffResults)
  }

  private def loadStages(json: JValue, wfOpt: Option[OpWorkflow], path: String): Try[Array[OPStage]] = {
    wfOpt.map(wf => loadStages(json, Stages, wf, path)).getOrElse(loadStages(json, Stages, path).map(_._1))
  }

  private def loadStages(json: JValue, field: FN, path: String): Try[(Array[OPStage], Array[OPFeature])] = Try {
    val stagesJs = (json \ field.entryName).extract[JArray].arr
    val (recoveredStages, recoveredFeatures) = ArrayBuffer.empty[OPStage] -> ArrayBuffer.empty[OPFeature]
    for {j <- stagesJs} {
      val stage = new OpPipelineStageReader(recoveredFeatures)
        .loadFromJson(j, path = path, asSpark = asSpark).asInstanceOf[OPStage]
      recoveredStages += stage
      recoveredFeatures += stage.getOutput()
    }
    recoveredStages.toArray -> recoveredFeatures.toArray
  }

  private def loadStages(json: JValue, field: FN, workflow: OpWorkflow, path: String): Try[Array[OPStage]] = Try {
    val generators = workflow.getRawFeatures().map(_.originStage)
    val stagesJs = (json \ field.entryName).extract[JArray].arr
    val recoveredStages = stagesJs.flatMap { j =>
      val stageUid = (j \ Uid.entryName).extract[String]
      val originalStage = workflow.getStages().find(_.uid == stageUid)
      originalStage match {
        case Some(os) => Option(
          new OpPipelineStageReader(os).loadFromJson(j, path = path, asSpark = asSpark)).map(_.asInstanceOf[OPStage]
        )
        case None if generators.exists(_.uid == stageUid) => None // skip the generator since they are in the workflow
        case None => throw new RuntimeException(s"Workflow does not contain a stage with uid: $stageUid")
      }
    }
    generators ++ recoveredStages
  }

  private def resolveFeatures(json: JValue, stages: Array[OPStage]): Try[Array[OPFeature]] = Try {
    val featuresArr = (json \ AllFeatures.entryName).extract[JArray].arr
    val stagesMap = stages.map(stage => stage.uid -> stage).toMap[String, OPStage]

    // should have been serialized in topological order
    // so that parent features can be used to construct each new feature
    val featuresMap = featuresArr.foldLeft(Map.empty[String, OPFeature])((featMap, feat) =>
      FeatureJsonHelper.fromJson(feat, stagesMap, featMap) match {
        case Failure(e) => throw new RuntimeException(s"Error resolving feature: $feat", e)
        case Success(f) => featMap + (f.uid -> f)
      }
    )

    // set input features to stages
    for {stage <- stages} {
      val inputIds = stage.getTransientFeatures().map(_.uid)
      val inFeatures = inputIds.map(id => TransientFeature(featuresMap(id))) // features are order dependent
      stage.set(stage.inputFeatures, inFeatures)
    }
    featuresMap.values.toArray
  }

  private def resolveResultFeatures(json: JValue, features: Array[OPFeature]): Try[Array[OPFeature]] = Try {
    val resultIds = (json \ ResultFeaturesUids.entryName).extract[Array[String]].toSet
    features.filter(f => resultIds.contains(f.uid))
  }

  private def resolveBlocklist
  (
    json: JValue,
    wfOpt: Option[OpWorkflow],
    features: Array[OPFeature],
    path: String
  ): Try[Array[OPFeature]] = {
    // For backward compatibility. The relevant field name is determined
    // by the max length of the blocklist found for each name.
    val potentialNames = Seq(
      (json \ BlocklistedFeaturesUids.entryName, BlocklistedStages),
      (json \ OldBlocklistedFeaturesUids.entryName, OldBlocklistedStages)
    )
    potentialNames.map { names =>
      if (names._1 != JNothing) { // for backwards compatibility
        for {
          feats <- wfOpt
            .map(wf => Success(wf.getAllFeatures() ++ wf.getBlocklist()))
            .getOrElse(loadStages(json, names._2, path).map(_._2))
          allFeatures = features ++ feats
          blocklistIds = names._1.extract[Array[String]]
        } yield blocklistIds.flatMap(uid => allFeatures.find(_.uid == uid))
      } else {
        Success(Array.empty[OPFeature])
      }
    }.maxBy(_.getOrElse(Array()).length)
  }

  private def resolveBlocklistMapKeys(json: JValue): Try[Map[String, Set[String]]] = Try {
    // For backward compatibility we combine new and deprecated keys.
    Seq(json \ BlocklistedMapKeys.entryName, json \ OldBlocklistedMapKeys.entryName)
      .flatMap(_.extractOpt[Map[String, List[String]]])
      .flatMap(_.map { case (k, vs) => k -> vs.toSet }).toMap
  }

  private def resolveRawFeatureFilterResults(json: JValue): Try[RawFeatureFilterResults] = {
    if ((json \ RawFeatureFilterResultsFieldName.entryName) != JNothing) {
      val resultsString = (json \ RawFeatureFilterResultsFieldName.entryName).extract[String]
      RawFeatureFilterResults.fromJson(resultsString)
    }
    else { // for backwards compatibility
      /**
       * RawFeatureDistributions is now contained in and written / read through RawFeatureFilterResults.
       * All setters of RawFeatureDistributions are now deprecated.
       * This resolve function is to allow backwards compatibility where RawFeatureDistributions was a saved field
       */
      val rawFeatureDistributionsEntryName = "rawFeatureDistributions"
      if ((json \ rawFeatureDistributionsEntryName) != JNothing) {
        val distString = (json \ rawFeatureDistributionsEntryName).extract[String]
        FeatureDistribution.fromJson(distString).map(d => RawFeatureFilterResults(rawFeatureDistributions = d))
      } else {
        Success(RawFeatureFilterResults())
      }
    }
  }

}

private object FileReader {

  def loadFile(pathString: String)(implicit conf: Configuration): String = {
    val path = new Path(pathString)
    val fs = path.getFileSystem(conf)
    val allFiles = fs.listFiles(path, false)
    var partPath: Option[Path] = None
    while (allFiles.hasNext) {
      val p = allFiles.next().getPath
      if (p.getName.startsWith("part")) {
        partPath = Option(p)
      }
    }
    val finalPath = partPath.getOrElse(path)
    val codecFactory = new CompressionCodecFactory(conf)
    val codec = Option(codecFactory.getCodec(finalPath))
    val in = fs.open(finalPath)
    val read = codec.map( c => Source.fromInputStream(c.createInputStream(in)).mkString )
      .getOrElse( IOUtils.toString(in, "UTF-8") )
    in.close()
    read
  }
}



