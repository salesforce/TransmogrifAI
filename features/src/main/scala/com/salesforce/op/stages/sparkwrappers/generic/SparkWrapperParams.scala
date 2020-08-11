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

package com.salesforce.op.stages.sparkwrappers.generic

import com.salesforce.op.features.FeatureSparkTypes
import com.salesforce.op.stages.{OpPipelineStageBase, SparkStageParam}
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.param.{Params, StringArrayParam}
import org.apache.spark.ml.{PipelineStage, Transformer}
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.{DataFrame, Row, SparkSession}


/**
 * Object to allow generic string based access to parameters of wrapped spark class
 *
 * @tparam S type of Spark stage to wrap
 */
trait SparkWrapperParams[S <: PipelineStage with Params] extends Params {
  self: OpPipelineStageBase =>

  final val sparkInputColParamNames = new StringArrayParam(
    parent = this,
    name = "sparkInputColParamNames",
    doc = "names of parameters that control input columns for Spark stage"
  )

  final val sparkOutputColParamNames = new StringArrayParam(
    parent = this,
    name = "sparkOutputColParamNames",
    doc = "names of parameters that control output columns for Spark stage"
  )

  final val sparkMlStage = new SparkStageParam[S](
    parent = this, name = SparkWrapperParams.SparkStageParamName,
    doc = "the spark stage that is being wrapped for TransmogrifAI"
  )

  setDefault(sparkMlStage, None)

  protected def setSparkMlStage(stage: Option[S]): this.type = {
    set(sparkMlStage, stage)
    this
  }

  /**
   * Method to access the spark stage being wrapped
   *
   * @return Option of spark ml stage
   */
  def getSparkMlStage(): Option[S] = $(sparkMlStage)

  /**
   * XGBoost model save requires a non-empty dataframe to save correctly with Mleap
   */
  private var outputDF: Option[DataFrame] = None

  def setOutputDF(df: DataFrame): Unit = {
    outputDF = Option(df)
    sparkMlStage.sbc = Option(SparkBundleContext().withDataset(df))
  }

  private[op] def getOutputDF(): Option[DataFrame] = outputDF

  /**
   * Sets a save path for wrapped spark stage
   *
   * @param path
   */
  def setStageSavePath(path: String): this.type = {
    sparkMlStage.savePath = Option(path)
    val ds = outputDF.getOrElse(getEmptyDF)
    sparkMlStage.sbc = Option(SparkBundleContext().withDataset(ds))
    this
  }


  /**
   * Used for saving models in ML leap format
   */
  private def getEmptyDF: DataFrame = {
    val rawSchema = FeatureSparkTypes.toStructType(getInputFeatures(): _*)
    val spark = SparkSession.active
    val df = spark.emptyDataset[Row](RowEncoder(rawSchema))
    getSparkMlStage().collect { case t: Transformer => t }.foldLeft(df) { case (d, t) => t.transform(d) }
  }

  /**
   * Gets a save path for wrapped spark stage
   */
  def getStageSavePath(): Option[String] = sparkMlStage.savePath

  /**
   * Gets names of parameters that control input columns for Spark stage
   */
  def getInputColParamNames(): Array[String] = $(sparkInputColParamNames)

  /**
   * Gets names of parameters that control output columns for Spark stage
   */
  def getOutputColParamNames(): Array[String] = $(sparkOutputColParamNames)
}

object SparkWrapperParams {
  val SparkStageParamName = "sparkMlStage"
}
