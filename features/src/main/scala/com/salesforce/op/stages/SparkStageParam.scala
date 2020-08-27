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

import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import ml.combust.bundle.BundleFile
import ml.combust.bundle.dsl.Bundle
import ml.combust.bundle.serializer.SerializationFormat
import ml.combust.mleap.spark.SparkSupport._
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.bundle.SparkBundleContext
import ml.combust.mleap.runtime.MleapSupport._
import ml.combust.mleap.runtime.frame.{Transformer => MLeapTransformer}
import org.apache.spark.ml.param.{Param, ParamPair, Params}
import org.apache.spark.ml.util.{Identifiable, MLReader, MLWritable}
import org.apache.spark.ml.{PipelineStage, Transformer}
import org.apache.spark.util.SparkUtils
import org.json4s.JsonAST.{JBool, JObject, JValue}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{compact, parse, render}
import org.json4s.{DefaultFormats, Formats, JString}
import resource._

import scala.util.{Failure, Success, Try}

class SparkStageParam[S <: PipelineStage with Params]
(
  parent: String,
  name: String,
  doc: String,
  isValid: Option[S] => Boolean
) extends Param[Option[S]](parent, name, doc, isValid) {

  import SparkStageParam._

  /**
   * Spark stage saving path
   */
  var savePath: Option[String] = None
  @transient var sbc: Option[SparkBundleContext] = None
  var localTransformer: Option[MLeapTransformer] = None

  def this(parent: String, name: String, doc: String) =
    this(parent, name, doc, (_: Option[S]) => true)

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
   */
  override def jsonEncode(sparkStage: Option[S]): String = {
    def json(className: String, uid: String) = compact(render(("className" -> className) ~ ("uid" -> uid)))
    (sparkStage, savePath, sbc) match {
      case (Some(stage), Some(p), Some(c)) =>
        for {bundle <- managed(BundleFile(s"file:$p/${stage.uid}"))} {
          stage.asInstanceOf[Transformer].writeBundle.format(SerializationFormat.Json).save(bundle)(c).get
        }
        json(className = stage.getClass.getName, uid = stage.uid)
      case (Some(stage), Some(p), None) =>
        val stagePath = new Path(p, stage.uid).toString
        stage.asInstanceOf[MLWritable].write.save(stagePath)
        json(className = stage.getClass.getName, uid = stage.uid)
      case (Some(s), None, _) =>
        throw new RuntimeException(s"Path must be set before Spark stage '${s.uid}' can be saved")
      case _ =>
        json(className = NoClass, uid = NoUID)
    }
  }

  private def getPathUid(jsonStr: String): (Option[String], Option[String], Option[Boolean]) = {
    val json = parse(jsonStr)
    val uid = (json \ "uid").extractOpt[String]
    val path = (json \ "path").extractOpt[String]
    val asSpark = (json \ "asSpark").extractOpt[Boolean]
    (path, uid, asSpark)
  }


  /**
   * Decodes the saved path string and uses the Pipeline.load method
   * to recover the stage.
   */
  override def jsonDecode(jsonStr: String): Option[S] = {
    val dirBundle: Option[Either[S, MLeapTransformer]] = jsonDecodeMLleap(jsonStr)
    println(dirBundle.get)
    dirBundle.flatMap{
      case Right(mleap) =>
        localTransformer = Option(mleap)
        None
      case Left(spark) => Option(spark)
    }.orElse { // for backwards compatibility
      getPathUid(jsonStr) match {
        case (_, Some(NoUID), _) => None
        case (Some(p), Some(stageUid), Some(true)) =>
          val stagePath = new Path(p, stageUid).toString
          val json = parse(jsonStr)
          val className = (json \ "className").extract[String]
          val cls = SparkUtils.classForName(className)
          val stage = cls.getMethod("read").invoke(null).asInstanceOf[MLReader[PipelineStage]].load(stagePath)
          Option(stage).map(_.asInstanceOf[S])
        case _ => None
      }
    }
  }

  private def loadError(loaded: Try[Bundle[_]]): Bundle[_] = {
    loaded match {
      case Failure(exception) =>
        throw new Exception(s"Failed to load model because of: $exception")
      case Success(mod) => mod
    }
  }


  /**
   * Decodes the saved path string and uses the Pipeline.load method
   * to recover the stage as an Mleap transformer
   */
  def jsonDecodeMLleap(jsonStr: String): Option[Either[S, MLeapTransformer]] = {
    println(getPathUid(jsonStr))
    getPathUid(jsonStr) match {
      case (None, _, _) | (_, None, _) | (_, Some(NoUID), _) =>
        savePath = None
        None
      case (Some(p), Some(stageUid), asSpark) =>
        savePath = Option(p)
        val loaded = for {bundle <- managed(BundleFile(s"file:$p/$stageUid"))} yield {
          if (asSpark.getOrElse(true)) Left(loadError(bundle.loadSparkBundle()).root.asInstanceOf[S])
          else {
            val tryload = bundle.loadMleapBundle()
            println(tryload)
            println(tryload.isFailure)
            println(tryload.failed.get.getStackTrace.mkString("\n"))
            Right(loadError(bundle.loadMleapBundle()).root.asInstanceOf[MLeapTransformer])
          }
        }
        println(loaded.opt)
        loaded.opt
    }
  }
}


object SparkStageParam {
  implicit val formats: Formats = DefaultFormats
  val NoClass = ""
  val NoUID = ""

  def updateParamsMetadataWithPath(jValue: JValue, path: String, asSpark: Boolean): JValue = jValue match {
    case JObject(pairs) => JObject(
      pairs.map {
        case (SparkWrapperParams.SparkStageParamName, j) =>
          SparkWrapperParams.SparkStageParamName -> j
            .merge(JObject("path" -> JString(path)))
            .merge(JObject("asSpark" -> JBool(asSpark)))
        case param => param
      }
    )
    case j => throw new IllegalArgumentException(s"Cannot recognize JSON Spark params metadata: $j")
  }

}
