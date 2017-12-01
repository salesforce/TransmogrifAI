/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.utils.text._

import scala.reflect.runtime.universe.TypeTag

/**
 * Transformer that detects the language of the text
 *
 * @param detector a language detector instance (defaults to [[OptimaizeLanguageDetector]]
 * @param uid      uid of the stage
 */
class LangDetector[T <: Text]
(
  val detector: LanguageDetector = LangDetector.DefaultDetector,
  uid: String = UID[LangDetector[_]]
)(implicit tti: TypeTag[T])
  extends UnaryTransformer[T, RealMap](operationName = "langDet", uid = uid) {

  /**
   * Function used to convert input to output
   */
  override def transformFn: T => RealMap = text => {
    if (text.isEmpty) RealMap.empty
    else {
      val langs = detector.detectLanguages(text.v.get)
      langs.map { case (lang, confidence) => lang.entryName -> confidence }.toMap.toRealMap
    }
  }

}

object LangDetector {
  val DefaultDetector: LanguageDetector = new OptimaizeLanguageDetector()
}
