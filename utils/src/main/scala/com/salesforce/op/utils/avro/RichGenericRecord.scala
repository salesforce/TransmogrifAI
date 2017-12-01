/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.avro

import org.apache.avro.generic.GenericRecord
import scala.collection.JavaConverters._


import scala.util.Try

object RichGenericRecord {

  /** This class is a wrapper that helps accessing fields in Avro Generic records.
   *
   *
   * @param r The record the function is called on
   */
  implicit class RichGenericRecord(val r: GenericRecord) extends AnyVal {


    /** Assumes field is a standard field
     *
     * @param fieldName name of the field
     * @tparam T return type
     * @return
     */
    def getValue[T](fieldName: String): Option[T] = {
      // Check that field exists in schema
      require(Option(r.getSchema.getField(fieldName)).isDefined,
        s"${fieldName} is not found in Avro schema!")

      val field = Option(r.get(fieldName))
      (field map {
        case r: GenericRecord => r.get("value")
        case r => r
      }).map(x => javaConvert(x).asInstanceOf[T])
    }
  }

  /** Converts types returned from avro into scala types
   *
   * @param in
   * @return
   */
  private def javaConvert(in: Any): Any = {
    in match {
      case s: java.lang.String => s
      case s: org.apache.avro.util.Utf8 => s.toString
      case i: java.lang.Integer => i.toInt
      case d: java.lang.Double => d.toDouble
      case l: java.lang.Long => l.toLong
      case b: java.lang.Boolean => b
      case f: java.lang.Float => f.toFloat
      case s: java.lang.Short => s.toShort
      case c: java.lang.Character => c.toChar
      case x => throw new NotImplementedError(s"${x.getClass} is not an implemented type")
    }
  }

}
