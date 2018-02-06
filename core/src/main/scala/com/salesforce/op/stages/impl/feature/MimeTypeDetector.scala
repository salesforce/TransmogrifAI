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
import org.apache.spark.ml.param.{Param, Params}
import org.apache.tika.detect.{DefaultDetector, Detector}
import org.apache.tika.metadata.{HttpHeaders, Metadata}
import org.apache.tika.mime.MediaType


/**
 * Detects MIME type for Base64 encoded binary data.
 */
class MimeTypeDetector(uid: String = UID[MimeTypeDetector])
  extends UnaryTransformer[Base64, Text](operationName = "mimeDetect", uid = uid)
    with MimeTypeDetectorParams {

  def transformFn: Base64 => Text = in =>
    TikaHelper.detect(in, $(maxBytesToParse), $(typeHint)).map(_.toString).toText

}

/**
 * Detects MIME type for Base64Map encoded binary data.
 */
class MimeTypeMapDetector(uid: String = UID[MimeTypeMapDetector])
  extends UnaryTransformer[Base64Map, PickListMap](operationName = "mimeMapDetect", uid = uid)
    with MimeTypeDetectorParams {

  def transformFn: Base64Map => PickListMap = in => {
    val maxBytes = $(maxBytesToParse)
    val tHint = $(typeHint)

    in.value
      .mapValues(v => TikaHelper.detect(v.toBase64, maxBytes, tHint))
      .collect { case (k, Some(v)) => k -> v.toString }.toPickListMap
  }

}

/**
 * Params for MIME type detection for Base64 encoded binary data.
 */
private[op] trait MimeTypeDetectorParams extends Params {
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
}

/**
 * Tika helper
 */
object TikaHelper {
  private val detector: Detector = new DefaultDetector()
  private val emptyMeta = new Metadata()

  /**
   * Detects the content type of the given input document
   *
   * @param in              input document
   * @param maxBytesToParse maximum number of bytes to parse during detection
   * @param typeHint        MIME type hint, i.e. 'application/json', 'text/plain' etc.
   * @throws IOException if the document input stream could not be read
   * @return returns application/octet-stream if the type of the document can not be detected, or the detected type.
   */
  def detect(in: Base64, maxBytesToParse: Long, typeHint: String): Option[MediaType] =
    in.mapInputStream(v => detect(new BoundedInputStream(v, maxBytesToParse), typeHint))

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
