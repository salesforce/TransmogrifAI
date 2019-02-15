/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.utils.geo

import java.nio.file.Paths

import com.salesforce.op.test.TestCommon
import org.apache.log4j.Level
import org.apache.lucene.store.{BaseDirectory, NIOFSDirectory, RAMDirectory, SimpleFSDirectory}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

import scala.util.{Failure, Success}

@RunWith(classOf[JUnitRunner])
class LuceneReverseGeocoderTest extends FlatSpec with TestCommon {
  loggingLevel(Level.INFO)

  import LuceneReverseGeocoder._

  val sampleSize = 1000
  lazy val cities = loadWorldCitiesData(Paths.get(dataDirectory("data"))).take(sampleSize)

  // val indexDirectory: BaseDirectory = new NIOFSDirectory(Paths.get("/tmp/world-cities"))
  val indexDirectory: BaseDirectory = new RAMDirectory()

  val geocoder = new LuceneReverseGeocoder()
  lazy val index = openIndex(indexDirectory) match {
    case Failure(err) => fail(err)
    case Success(idx) => idx
  }

  Spec[LuceneReverseGeocoder] should "build index of cities" in {
    buildIndex(cities, indexDirectory) match {
      case Failure(err) => fail(err)
      case Success(elapsed) =>
        elapsed should be > 0L
        indexDirectory.listAll().length should be > 0
    }
  }
  it should "open an index" in {
    index.collectionStatistics("id").maxDoc() shouldBe cities.size
  }
  it should "get nearest cities" in {
    val city = cities.head
    val results = geocoder.nearestCities(index,
      latitude = city.latitude, longitude = city.longitude,
      radiusInKM = 10, numOfResults = 10)

    results should contain(city)
  }
  it should "get nearest countries" in {
    val city = cities.head
    val results = geocoder.nearestCountries(index,
      latitude = city.latitude, longitude = city.longitude,
      radiusInKM = 10, numOfResults = 10)

    results should contain(city.country)
  }

  it should "get nearest cities in a timely fashion" in {
    val start = System.currentTimeMillis()
    val items = cities.take(10000)
    for { city <- items } {
      val results = geocoder.nearestCountries(index,
        latitude = city.latitude, longitude = city.longitude,
        radiusInKM = 10, numOfResults = 10)

      results should contain(city.country)
    }
    val elapsed = System.currentTimeMillis() - start
    println((elapsed.toDouble / items.size) + "ms per query")
  }

  //  it should "nearest cities to Palo Alto, CA" in {
  //    openIndex(indexDirectory) match {
  //      case Failure(err) => fail(err)
  //      case Success(index) =>
  //        val results = geocoder.nearestCities(
  //          index, latitude = 37.4419, longitude = -122.1430, radiusInKM = 10, numOfResults = 10)
  //
  //        results.size should be >= 1
  //        // results should contain (city)
  //        // println("QUERY: " + city)
  //        results.foreach(println)
  //    }
  //  }
}
