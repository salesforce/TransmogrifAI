/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages

import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.Metadata


/**
 * Note this should be removed as a param and changed to a var if move stage reader and writer into op
 * and out of ml. Is currently a param to prevent having the setter method be public.
 */
private[stages] class MetadataParam(parent: String, name: String, doc: String, isValid: Metadata => Boolean)
  extends Param[Metadata](parent, name, doc, isValid) {

  def this(parent: String, name: String, doc: String) =
    this(parent, name, doc, (_: Metadata) => true)

  def this(parent: Identifiable, name: String, doc: String, isValid: Metadata => Boolean) =
    this(parent.uid, name, doc, isValid)

  def this(parent: Identifiable, name: String, doc: String) = this(parent.uid, name, doc)

  /** Creates a param pair with the given value (for Java). */
  override def w(value: Metadata): ParamPair[Metadata] = super.w(value)

  override def jsonEncode(value: Metadata): String = value.json

  override def jsonDecode(json: String): Metadata = Metadata.fromJson(json)
}
