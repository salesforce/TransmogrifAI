/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types

class TextList(val value: Seq[String]) extends OPList[String] {
  def this(v: String*)(implicit d: DummyImplicit) = this(v)
}
object TextList {
  def apply(value: Seq[String]): TextList = new TextList(value)
  def empty: TextList = FeatureTypeDefaults.TextList
}

class DateList(val value: Seq[Long]) extends OPList[Long] {
  def this(v: Long*)(implicit d: DummyImplicit) = this(v)
}
object DateList {
  def apply(value: Seq[Long]): DateList = new DateList(value)
  def empty: DateList = FeatureTypeDefaults.DateList
}

class DateTimeList(val value: Seq[Long]) extends OPList[Long] {
  def this(v: Long*)(implicit d: DummyImplicit) = this(v)
}
object DateTimeList {
  def apply(value: Seq[Long]): DateTimeList = new DateTimeList(value)
  def empty: DateTimeList = FeatureTypeDefaults.DateTimeList
}

