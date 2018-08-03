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

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.{PipelineStage, SparkDefaultParamsReadWrite}
import org.apache.spark.ml.param.{Param, ParamPair, Params}
import org.apache.spark.ml.util.{Identifiable, MLReader, MLWritable}
import org.apache.spark.util.SparkUtils
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{compact, parse, render}
import org.json4s.{DefaultFormats, Formats}

object SparkStageParam {
  final val NoPath: String = ""
}

class SparkStageParam[S <: PipelineStage with Params]
(
  parent: String,
  name: String,
  doc: String,
  isValid: Option[S] => Boolean
) extends Param[Option[S]](parent, name, doc, isValid) {
  @transient implicit val formats: Formats = DefaultFormats

  var savePath: Option[String] = None

  def this(parent: String, name: String, doc: String) =
    this(parent, name, doc, (v: Option[S]) => true)

  def this(parent: Identifiable, name: String, doc: String, isValid: Option[S] => Boolean) =
    this(parent.uid, name, doc, isValid)

  def this(parent: Identifiable, name: String, doc: String) = this(parent.uid, name, doc)

  /** Creates a param pair with the given value (for Java). */
  override def w(value: Option[S]): ParamPair[Option[S]] = super.w(value)

  /**
   * The Stage will be saved by creating a dummy Pipeline and using it's Writer. The
   * param itself will only encode the path, not the stage.
   *
   * If Stage is not set path will be serialized as an empty string
   *
   * @param value
   * @return
   */
  override def jsonEncode(value: Option[S]): String = {
    var path: Option[String] = None
    var className: Option[String] = None
    if (value.isDefined) {
      if (savePath.isEmpty) new RuntimeException("Path must be set before Spark Stage can be saved")

      val stage = value.get.asInstanceOf[PipelineStage]
      path = Option(new Path(savePath.get, stage.uid).toString)
      stage.asInstanceOf[MLWritable].write.save(path.get)
      className = Option(stage.getClass.getName)
    }
    compact(render(
      ("path" -> path.getOrElse(SparkStageParam.NoPath)) ~
        ("className" -> className.getOrElse(""))
    ))
  }

  /**
   * Decodes the saved path string and uses the Pipeline.load method
   * to recover the stage.
   *
   * @param jsonStr
   * @return
   */
  override def jsonDecode(jsonStr: String): Option[S] = {
    val json = parse(jsonStr)
    val path = (json \ "path").extract[String]
    if (path == SparkStageParam.NoPath){
      savePath = None
      None
    }
    else {
      savePath = Option(path)
      val cls = SparkUtils.classForName((json \ "className").extract[String])
      val stage = cls.getMethod("read").invoke(null).asInstanceOf[MLReader[PipelineStage]].load(path)
      Option(stage).map(_.asInstanceOf[S])
    }
  }
}
