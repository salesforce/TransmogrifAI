/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature


import java.io.InputStream

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import org.apache.commons.io.input.BoundedInputStream
import org.apache.spark.ml.param.Param
import org.apache.tika.detect.{DefaultDetector, Detector}
import org.apache.tika.metadata.{HttpHeaders, Metadata}
import org.apache.tika.mime.MediaType

/**
 * Detects MIME type for Base64 encoded binary data.
 */
class MimeTypeDetector(uid: String = UID[MimeTypeDetector])
  extends UnaryTransformer[Base64, Text](operationName = "mimeDetect", uid = uid) {

  final val typeHint = new Param[String](
    parent = this, name = "typeHint", doc = "MIME type hint, i.e. 'application/json', 'text/plain' etc.",
    isValid = (s: String) => s.isEmpty || MediaType.parse(s) != null
  )
  def setTypeHint(value: String): this.type = set(typeHint, value)

  final val maxBytesToParse = new Param[Long](
    parent = this, name = "maxBytesToParse", doc = "maximum number of bytes to parse during detection",
    isValid = (v: Long) => v >= 0L
  )
  def setMaxBytesToParse(value: Long): this.type = set(maxBytesToParse, value)

  setDefault(typeHint -> "", maxBytesToParse -> 1024L)

  def transformFn: Base64 => Text = _.mapInputStream(in => {
    val boundedIn = new BoundedInputStream(in, $(maxBytesToParse))
    TikaHelper.detect(boundedIn, $(typeHint)).toString
  }).toText

}

/**
 * Tika helper
 */
object TikaHelper {
  private val detector: Detector = new DefaultDetector()
  private val emptyMeta = new Metadata()

  /**
   * Detects the content type of the given input document.
   *
   * @param in       input stream
   * @param typeHint MIME type hint, i.e. 'application/json', 'text/plain' etc.
   * @throws IOException if the document input stream could not be read
   * @return returns application/octet-stream if the type of the document can not be detected, or the detected type.
   */
  def detect(in: InputStream, typeHint: String): MediaType = {
    val meta =
      if (typeHint == null || typeHint.isEmpty) emptyMeta
      else {
        val meta = new Metadata()
        meta.add(HttpHeaders.CONTENT_TYPE, typeHint)
        meta
      }
    // parses the input stream and detects the media type
    detector.detect(in, meta)
  }

}
