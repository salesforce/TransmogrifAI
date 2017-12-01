/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.date

import com.salesforce.op.test.TestCommon
import org.joda.time.{DateTime, DateTimeZone}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}


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
    val diff = DateTimeUtils.getRange(date.minusDays(numberOfDays).toString("YYYY/MM/dd"),
      date.toString("YYYY/MM/dd"))
    diff.length shouldBe numberOfDays + 1
  }

  it should "get date difference days from start date" in {
    val datePlusDays = DateTimeUtils.getDatePlusDays(now.toString("YYYY/MM/dd"), 31)
    datePlusDays shouldBe now.plusDays(31).toString("YYYY/MM/dd")
  }
}

