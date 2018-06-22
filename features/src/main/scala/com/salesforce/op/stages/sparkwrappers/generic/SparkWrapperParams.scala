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

package com.salesforce.op.stages.sparkwrappers.generic

import com.salesforce.op.stages.SparkStageParam
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.param.{Param, Params, StringArrayParam}


/**
 * Object to allow generic string based access to parameters of wrapped spark class
 *
 * @tparam S type of spark object to wrap
 */
trait SparkWrapperParams[S <: PipelineStage with Params] extends Params {
  self: PipelineStage =>

  final val sparkInputColParamNames = new StringArrayParam(
    parent = this,
    name = "sparkInputColParamNames",
    doc = "names of parameters that control input columns for spark stage"
  )

  final val sparkOutputColParamNames = new StringArrayParam(
    parent = this,
    name = "sparkOutputColParamNames",
    doc = "names of parameters that control output columns for spark stage"
  )

  /**
   * this must be private so that the stage can have it's path set properly
   */
  final val savePath = new Param[String](
    parent = this, name = "savePath", doc = "path to save the spark stage"
  )

  setDefault(savePath, SparkStageParam.NoPath)

  /**
   * this must be private so that the stage can have it's path set properly
   */
  final val sparkMlStage = new SparkStageParam[S](
    parent = this, name = "sparkMlStage", doc = "the spark stage that is being wrapped for optimus prime"
  )

  setDefault(sparkMlStage, None)

  def setSavePath(path: String): this.type = {
    set(savePath, path)
    sparkMlStage.savePath = Option($(savePath))
    this
  }

  protected def setSparkMlStage(stage: Option[S]): this.type = {
    set(sparkMlStage, stage)
    sparkMlStage.savePath = Option($(savePath))
    this
  }

  def getSavePath(): String = $(savePath)

  /**
   * Method to set spark parameters by string. Note will not set inputs or outputs as those should be taken from
   * the input features and output name respectively
   *
   * @param paramName string name of parameter want to set
   * @param value     value want to give parameter
   * @return spark stage with parameter set correctly
   */
  def setSparkParams(paramName: String, value: Any): this.type = {
    val sparkStage = if ($(sparkInputColParamNames).contains(paramName) ||
      $(sparkOutputColParamNames).contains(paramName)) {
      log.warn("Cannot set spark transformer inputs or outputs directly! " +
        "Values are set with pipeline stage setInputs method or the outputName method! \n" +
        s"Command to set $paramName to $value was ignored")
      $(sparkMlStage)
    } else if ($(sparkMlStage).exists(_.hasParam(paramName))) {
      $(sparkMlStage).map { s =>
        val param = s.getParam(paramName)
        s.set(param, value)
      }
    } else {
      log.warn("Invalid spark transformer param name! \n" +
        s"Command to set $paramName to $value was ignored")
      $(sparkMlStage)
    }
    set(sparkMlStage, sparkStage)
  }

  /**
   * Method to get the value of spark parameters by name
   *
   * @param paramName name of parameter want the value of
   * @return the value of the specified parameter
   */
  def getSparkParams(paramName: String): Option[Any] = {
    if ($(sparkMlStage).exists(_.hasParam(paramName))) {
      $(sparkMlStage).flatMap { s =>
        val param = s.getParam(paramName)
        s.get(param)
      }
    } else {
      log.warn("Invalid spark transformer param name! \n" +
        s"Command to get $paramName was ignored")
      None
    }
  }

  /**
   * Method to access the spark stage being wrapped
   *
   * @return Option of spark ml stage
   */
  def getSparkMlStage(): Option[S] = $(sparkMlStage)

  /**
   * for testing only
   */
  private[op] def getStageSavePath(): Option[String] = sparkMlStage.savePath
}
