/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.features.types

/**
 * A list of text values
 *
 * @param value list of text values
 */
class TextList(val value: Seq[String]) extends OPList[String] {
  def this(v: String*)(implicit d: DummyImplicit) = this(v)
}
object TextList {
  def apply(value: Seq[String]): TextList = new TextList(value)
  def empty: TextList = FeatureTypeDefaults.TextList
}

/**
 * A list of date values
 *
 * @param value list of date values (values assumed to be in ms since Epoch)
 */
class DateList(val value: Seq[Long]) extends OPList[Long] {
  def this(v: Long*)(implicit d: DummyImplicit) = this(v)
}
object DateList {
  def apply(value: Seq[Long]): DateList = new DateList(value)
  def empty: DateList = FeatureTypeDefaults.DateList
}

/**
 * A list of date & time values
 *
 * @param value list of date & time values (values assumed to be in ms since Epoch)
 */
class DateTimeList(value: Seq[Long]) extends DateList(value) {
  def this(v: Long*)(implicit d: DummyImplicit) = this(v)
}
object DateTimeList {
  def apply(value: Seq[Long]): DateTimeList = new DateTimeList(value)
  def empty: DateTimeList = FeatureTypeDefaults.DateTimeList
}

