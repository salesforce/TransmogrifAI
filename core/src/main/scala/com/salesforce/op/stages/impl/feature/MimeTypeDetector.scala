/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature


import java.io.InputStream

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import org.apache.spark.ml.param.Param
import org.apache.tika.detect.{DefaultDetector, Detector}
import org.apache.tika.metadata.{HttpHeaders, Metadata}
import org.apache.tika.mime.MediaType

/**
 * Detects MIME type for Base64 encoded binary data
 */
class MimeTypeDetector(uid: String = UID[MimeTypeDetector])
  extends UnaryTransformer[Base64, Text](operationName = "mimeDetect", uid = uid) {

  final val typeHint = new Param[String](
    parent = this, name = "typeHint", doc = "MIME type hint, i.e. 'application/json', 'text/plain' etc."
  )
  setDefault(typeHint, "")

  def setTypeHint(value: String): this.type = set(typeHint, value)

  def transformFn: Base64 => Text = (b: Base64) => {
    val mimeType = b.asInputStream.map(in =>
      try MimeTypeDetector.detect(in, $(typeHint)).toString finally in.close()
    )
    mimeType.toText
  }

}

private object MimeTypeDetector {
  private val detector: Detector = new DefaultDetector()
  private val emptyMeta = new Metadata()

  def detect(in: InputStream, typeHint: String): MediaType = {
    val meta = if (typeHint != null && typeHint.length > 0) {
      val m = new Metadata()
      m.add(HttpHeaders.CONTENT_TYPE, typeHint)
      m
    } else emptyMeta

    detector.detect(in, meta)
  }
}
