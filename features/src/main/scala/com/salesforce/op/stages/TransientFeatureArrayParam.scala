/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages

import com.salesforce.op.features._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.json4s.DefaultFormats
import org.json4s.JsonAST.{JArray, JValue}
import org.json4s.jackson.JsonMethods.{compact, parse, render}

import scala.util.{Failure, Success}


/**
 * Transient array param for storing input features for stages
 */
private[stages] class TransientFeatureArrayParam
(
  parent: String,
  name: String,
  doc: String,
  isValid: Array[TransientFeature] => Boolean
) extends Param[Array[TransientFeature]](parent, name, doc, isValid) {

  @transient implicit val formats = DefaultFormats

  def this(parent: String, name: String, doc: String) =
    this(parent, name, doc, (_: Array[TransientFeature]) => true)

  def this(parent: Identifiable, name: String, doc: String, isValid: Array[TransientFeature] => Boolean) =
    this(parent.uid, name, doc, isValid)

  def this(parent: Identifiable, name: String, doc: String) = this(parent.uid, name, doc)

  /** Creates a param pair with the given value (for Java). */
  override def w(value: Array[TransientFeature]): ParamPair[Array[TransientFeature]] = super.w(value)

  override def jsonEncode(value: Array[TransientFeature]): String = {
    compact(render(JArray(value.map(_.toJson).toList)))
  }

  override def jsonDecode(json: String): Array[TransientFeature] = {
    parse(json).extract[Array[JValue]].map(obj => {
      TransientFeature(obj) match {
        case Failure(e) => throw new RuntimeException("Failed to parse TransientFeature", e)
        case Success(v) => v
      }
    })
  }
}
