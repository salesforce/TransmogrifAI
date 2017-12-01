/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages

import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{DataType, StructType}


/**
 * Note this should be removed as a param and changed to a var if move stage reader and writer into op
 * and out of ml. Is currently a param to prevent having the setter method be public.
 */
private[stages] class SchemaParam(parent: String, name: String, doc: String, isValid: StructType => Boolean)
  extends Param[StructType](parent, name, doc, isValid) {

  def this(parent: String, name: String, doc: String) =
    this(parent, name, doc, (_: StructType) => true)

  def this(parent: Identifiable, name: String, doc: String, isValid: StructType => Boolean) =
    this(parent.uid, name, doc, isValid)

  def this(parent: Identifiable, name: String, doc: String) = this(parent.uid, name, doc)

  /** Creates a param pair with the given value (for Java). */
  override def w(value: StructType): ParamPair[StructType] = super.w(value)

  override def jsonEncode(value: StructType): String = value.json

  override def jsonDecode(json: String): StructType = DataType.fromJson(json).asInstanceOf[StructType]
}
