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

import com.salesforce.op.utils.reflection.ReflectionUtils
import org.json4s.JValue
import org.json4s.JsonAST.{JObject, JString}

import scala.reflect.ClassTag
import scala.util.Try


/**
 * Default value reader/writer implementation used to (de)serialize stage arguments from/to trained models
 * based on their class name and no args ctor.
 *
 * @param valueName value name
 * @tparam T value type to read/write
 */
final class DefaultValueReaderWriter[T <: AnyRef](valueName: String)(implicit val ct: ClassTag[T])
  extends ValueReaderWriter[T] with OpPipelineStageReadWriteFormats with OpPipelineStageSerializationFuns {

  /**
   * Read value from json
   *
   * @param valueClass value class
   * @param json       json to read argument value from
   * @return read result
   */
  def read(valueClass: Class[T], json: JValue): Try[T] = Try {
    val className = (json \ "className").extract[String]
    ReflectionUtils.newInstance[T](className)
  }

  /**
   * Write value to json
   *
   * @param value value to write
   * @return write result
   */
  def write(value: T): Try[JValue] = Try {
    val arg = serializeArgument(valueName, value)
    JObject("className" -> JString(arg.value.toString))
  }

}
