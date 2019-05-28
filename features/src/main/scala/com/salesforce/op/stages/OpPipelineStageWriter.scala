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

package com.salesforce.op.stages

import com.salesforce.op.stages.OpPipelineStageReaderWriter._
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.util.MLWriter
import org.apache.spark.ml.{Estimator, SparkDefaultParamsReadWrite}
import org.json4s.JsonAST.{JObject, JValue}
import org.json4s.jackson.JsonMethods.{compact, render}

import scala.util.{Failure, Success}

/**
 * MLWriter class used to write TransmogrifAI stages to disk
 *
 * @param stage a stage to save
 */
final class OpPipelineStageWriter(val stage: OpPipelineStageBase) extends MLWriter {

  override protected def saveImpl(path: String): Unit = {
    val metadataPath = new Path(path, "metadata").toString
    sc.parallelize(Seq(writeToJsonString(path)), 1).saveAsTextFile(metadataPath)
  }

  /**
   * Stage metadata json string
   *
   * @return stage metadata json string
   */
  def writeToJsonString(path: String): String = compact(writeToJson(path))

  /**
   * Stage metadata json
   *
   * @return stage metadata json
   */
  def writeToJson(path: String): JObject = {
    stage match {
      case _: Estimator[_] => return JObject() // no need to serialize estimators
      case s: SparkWrapperParams[_] =>
        // Set save path for all Spark wrapped stages of type [[SparkWrapperParams]] so they can save
        s.setStageSavePath(path)
      case _ =>
    }
    // We produce stage metadata for all the Spark params
    val metadata = SparkDefaultParamsReadWrite.getMetadataToSave(stage)

    // Write out the stage using the specified writer instance
    val writer = readerWriterFor[OpPipelineStageBase](stage.getClass.asInstanceOf[Class[OpPipelineStageBase]])
    val stageJson: JValue = writer.write(stage) match {
      case Failure(err) => throw new RuntimeException(s"Failed to write out stage '${stage.uid}'", err)
      case Success(json) => json
    }

    // Join metadata & with stage ctor args
    val j = metadata.merge(JObject(FieldNames.CtorArgs.entryName -> stageJson))
    render(j).asInstanceOf[JObject]
  }

}
