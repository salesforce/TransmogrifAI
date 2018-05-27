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

package com.salesforce.op.utils.date

import com.salesforce.op.test.TestCommon
import org.joda.time.{DateTime, DateTimeZone}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class DateTimeUtilsTest extends FlatSpec with TestCommon {

  val dateStr = "2017-03-29T14:00:07.000Z"
  val date = DateTime.parse(dateStr)
  val now = DateTime.now(DateTimeZone.UTC)

  Spec(DateTimeUtils.getClass) should "parse date in Iso format" in {
    DateTimeUtils.parse(dateStr) shouldBe date.getMillis
  }

  it should "parse date in yyyy-MM-dd HH:mm:ss.SSS format" in {
    val formattedStr = "2017-03-29 14:00:07.000"
    DateTimeUtils.parse(formattedStr) shouldBe date.getMillis
  }

  it should "parse Unix" in {
    DateTimeUtils.parseUnix(now.getMillis) shouldBe now.toString("YYYY/MM/dd")
  }

  it should "get range between two dates" in {
    val numberOfDays = 500
    val diff = DateTimeUtils.getRange(
      date.minusDays(numberOfDays).toString("YYYY/MM/dd"),
      date.toString("YYYY/MM/dd")
    )
    diff.length shouldBe numberOfDays + 1
  }

  it should "get date difference days from start date" in {
    val datePlusDays = DateTimeUtils.getDatePlusDays(now.toString("YYYY/MM/dd"), 31)
    datePlusDays shouldBe now.plusDays(31).toString("YYYY/MM/dd")
  }
}

