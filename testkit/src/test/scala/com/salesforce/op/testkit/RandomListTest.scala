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

package com.salesforce.op.testkit

import java.text.SimpleDateFormat

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestCommon
import com.salesforce.op.testkit.RandomList.{NormalGeolocation, UniformGeolocation}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec}

import scala.language.postfixOps


@RunWith(classOf[JUnitRunner])
class RandomListTest extends FlatSpec with TestCommon with Assertions {
  private val numTries = 10000
  private val rngSeed = 314159214142136L

  private def check[D, T <: OPList[D]](
    g: RandomList[D, T],
    minLen: Int, maxLen: Int,
    predicate: (D => Boolean) = (_: D) => true
  ) = {
    g reset rngSeed

    def segment = g limit numTries

    segment count (_.value.length < minLen) shouldBe 0
    segment count (_.value.length > maxLen) shouldBe 0
    segment foreach (list => list.value foreach { x =>
      predicate(x) shouldBe true
    })
  }

  private val df = new SimpleDateFormat("dd/MM/yy")

  Spec[Text, RandomList[String, TextList]] should "generate lists of strings" in {
    val sut = RandomList.ofTexts(RandomText.countries, 0, 4)
    check[String, TextList](sut, 0, 4, _.length > 0)

    (sut limit 7 map (_.value.toList)) shouldBe
      List(
        List("Madagascar", "Gondal", "Zephyria"),
        List("Holy Alliance"),
        List("North American Union"),
        List("Guatemala", "Estonia", "Kolechia"),
        List(),
        List("Myanmar", "Bhutan"),
        List("Equatorial Guinea")
      )
  }

  Spec[Date, RandomList[Long, DateList]] should "generate lists of dates" in {
    val dates = RandomIntegral.dates(df.parse("01/01/2017"), 1000, 1000000)
    val sut = RandomList.ofDates(dates, 11, 22)
    var d0 = 0L
    check[Long, DateList](sut, 11, 22, d => {
      val d1 = d0
      d0 = d
      d > d1
    })
  }

  Spec[DateTimeList, RandomList[Long, DateTimeList]] should "generate lists of datetimes" in {
    val datetimes = RandomIntegral.datetimes(df.parse("01/01/2017"), 1000, 1000000)
    val sut = RandomList.ofDateTimes(datetimes, 11, 22)
    var d0 = 0L
    check[Long, DateTimeList](sut, 11, 22, d => {
      val d1 = d0
      d0 = d
      d > d1
    })
  }

  Spec[UniformGeolocation] should "generate uniformly distributed geolocations" in {
    val sut = RandomList.ofGeolocations
    val segment = sut limit numTries
    segment foreach (_.value.length shouldBe 3)
  }

  Spec[NormalGeolocation] should "generate geolocations around given point" in {
    for {accuracy <- GeolocationAccuracy.values} {
      val geolocation = RandomList.ofGeolocationsNear(37.444136, 122.163160, accuracy)
      val segment = geolocation limit numTries
      segment foreach (_.value.length shouldBe 3)
    }
  }
}
