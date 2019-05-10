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

import com.salesforce.op.stages.OpPipelineStageReadWriteShared._
import com.salesforce.op.utils.reflection.ReflectionUtils
import org.json4s.JValue

import scala.util.Try

/**
 * Stage reader/writer implementation used to (de)serialize stages from/to trained models
 *
 * @tparam StageType stage type to read/write
 */
trait OpPipelineStageJsonReaderWriter[StageType <: OpPipelineStageBase] extends OpPipelineStageReadWriteFormats {

  /**
   * Read stage from json
   *
   * @param stageClass stage class
   * @param json       json to read stage from
   * @return read result
   */
  def read(stageClass: Class[StageType], json: JValue): Try[StageType]

  /**
   * Write stage to json
   *
   * @param stage stage instance to write
   * @return write result
   */
  def write(stage: StageType): Try[JValue]
}


private[op] trait SerializationFuns {

  def serializeArgument(argName: String, value: AnyRef): AnyValue = {
    try {
      val valueClass = value.getClass
      // Test that value has no external dependencies and can be constructed without ctor args or is an object
      ReflectionUtils.newInstance[AnyRef](valueClass.getName)
      AnyValue(AnyValueTypes.ClassInstance, valueClass.getName, Option(valueClass.getName))
    } catch {
      case error: Exception => throw new RuntimeException(
        s"Argument '$argName' [${value.getClass.getName}] cannot be serialized. " +
          s"Make sure ${value.getClass.getName} has either no-args ctor or is an object, " +
          "and does not have any external dependencies, e.g. use any out of scope variables.", error)
    }
  }

}
