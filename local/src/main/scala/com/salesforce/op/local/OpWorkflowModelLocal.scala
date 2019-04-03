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

package com.salesforce.op.local


import java.nio.file.Paths

import com.github.marschall.memoryfilesystem.MemoryFileSystemBuilder
import com.salesforce.op.OpWorkflowModel
import com.salesforce.op.features.FeatureSparkTypes
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.stages.{OPStage, OpTransformer}
import ml.combust.bundle.serializer.SerializationFormat
import ml.combust.bundle.{BundleContext, BundleRegistry}
import ml.combust.mleap.runtime.MleapContext
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable

/**
 * Enrichment for [[OpWorkflowModel]] to allow local scoring functionality
 */
trait OpWorkflowModelLocal extends Serializable {

  /**
   * Enrichment for [[OpWorkflowModel]] to allow local scoring functionality
   *
   * @param model [[OpWorkflowModel]]
   */
  implicit class RichOpWorkflowModel(model: OpWorkflowModel) {

    /**
     * Internal OP model representation
     *
     * @param output output name
     * @param model  model instance
     */
    private case class OPModel(output: String, model: OPStage with OpTransformer)

    /**
     * Internal MLeap model representation
     *
     * @param inputs  model inputs
     * @param output  model output
     * @param modelFn model function
     */
    private case class MLeapModel
    (
      inputs: Array[String],
      output: String,
      modelFn: Array[Any] => Any
    )

    /**
     * Prepares a score function for local scoring
     *
     * @param spark Spark Session needed for preparing scoring function,
     *              Once scoring function is returned the Spark Session can be shutdown
     *              since it's not required during local scoring.
     * @return score function for local scoring
     */
    def scoreFunction(implicit spark: SparkSession): ScoreFunction = {
      // Prepare the stages for scoring
      val stagesWithIndex = model.stages.zipWithIndex

      // Prepare an empty DataFrame with transformed schema & metadata (needed for loading MLeap models)
      val transformedData = makeTransformedDataFrame(model)

      // Collect all OP stages
      val opStages = stagesWithIndex.collect { case (s: OpTransformer, i) => OPModel(s.getOutputFeatureName, s) -> i }

      // Collect all Spark wrapped stages
      val sparkStages = stagesWithIndex.filterNot(_._1.isInstanceOf[OpTransformer]).collect {
        case (opStage: OPStage with SparkWrapperParams[_], i) if opStage.getSparkMlStage().isDefined =>
          val sparkStage = opStage.getSparkMlStage().get.asInstanceOf[Transformer]
          (opStage, sparkStage, i)
      }
      // Convert Spark wrapped stages into MLeap models
      val mleapStages = toMLeapModels(sparkStages, transformedData)

      // Combine all stages and apply the original order
      val allStages = (opStages ++ mleapStages).sortBy(_._2).map(_._1)
      val resultFeatures = model.getResultFeatures().map(_.name).toSet

      // Score Function
      input: Map[String, Any] => {
        val inputMap = mutable.Map.empty ++= input
        val transformedRow = allStages.foldLeft(inputMap) {
          // For OP Models we simply call transform
          case (row, OPModel(output, stage)) =>
            row += output -> stage.transformKeyValue(row.apply)

          // For MLeap models we call the prepared local model
          case (row, MLeapModel(inputs, output, modelFn)) =>
            val in = inputs.map(inputName => row.get(inputName) match {
              case None | Some(null) => null
              case Some(v) => v
            })
            row += output -> modelFn(in)
        }

        // Only return the result features of the model
        transformedRow.filterKeys(resultFeatures.contains).toMap
      }
    }

    /**
     * Prepares an empty DataFrame with transformed schema & metadata (needed for loading MLeap models)
     */
    private def makeTransformedDataFrame(model: OpWorkflowModel)(implicit spark: SparkSession): DataFrame = {
      val rawSchema = FeatureSparkTypes.toStructType(model.rawFeatures: _*)
      val df = spark.emptyDataset[Row](RowEncoder(rawSchema))
      model.stages.collect { case t: Transformer => t }.foldLeft(df) { case (d, t) => t.transform(d) }
    }

    /**
     * Convert Spark wrapped stages into MLeap local Models
     *
     * @param sparkStages     stages to convert
     * @param transformedData dataset with transformed schema & metadata (needed for loading MLeap models)
     * @return MLeap local stages
     */
    private def toMLeapModels
    (
      sparkStages: Seq[(OPStage, Transformer, Int)],
      transformedData: DataFrame
    ): Seq[(MLeapModel, Int)] = {
      // Setup a in-memory file system for MLeap model saving/loading
      val emptyPath = Paths.get("")
      val fs = MemoryFileSystemBuilder.newEmpty().build()

      // Setup two MLeap registries - one local and one for Spark
      val mleapRegistry = BundleRegistry("ml.combust.mleap.registry.default")
      val sparkRegistry = BundleRegistry("ml.combust.mleap.spark.registry.default")

      val sparkBundleContext = BundleContext[SparkBundleContext](
        SparkBundleContext(Option(transformedData), sparkRegistry),
        SerializationFormat.Json, sparkRegistry, fs, emptyPath
      )
      val mleapBundleContext = BundleContext[MleapContext](
        MleapContext(mleapRegistry), SerializationFormat.Json, mleapRegistry, fs, emptyPath)

      for {
        (opStage, sparkStage, i) <- sparkStages
      } yield try {
        val model = {
          // Serialize Spark model using Spark registry
          val opModel = sparkRegistry.opForObj[SparkBundleContext, AnyRef, AnyRef](sparkStage)
          val emptyModel = new ml.combust.bundle.dsl.Model(op = opModel.Model.opName)
          val serializedModel = opModel.Model.store(emptyModel, sparkStage)(sparkBundleContext)

          // Load MLeap model using MLeap local registry from the serialized model
          val mleapLocalModel = mleapRegistry.model[MleapContext, AnyRef](op = serializedModel.op)
          mleapLocalModel.load(serializedModel)(mleapBundleContext)
        }
        // Prepare and return MLeap model with inputs, output and model function
        MLeapModel(
          inputs = opStage.getTransientFeatures().map(_.name),
          output = opStage.getOutputFeatureName,
          modelFn = MLeapModelConverter.modelToFunction(model)
        ) -> i
      } catch {
        case e: Exception =>
          throw new RuntimeException(s"Failed to convert stage '${opStage.uid}' to MLeap stage", e)
      }
    }


  }

}
