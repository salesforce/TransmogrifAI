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

package com.salesforce.op.features.types

import java.nio.file.{Path, Paths}

import com.salesforce.op.test.TestCommon
import org.apache.lucene.document.{Document, NumericDocValuesField, StoredField}
import org.apache.lucene.index.{DirectoryReader, IndexWriter, IndexWriterConfig}
import org.apache.lucene.search._
import org.apache.lucene.spatial.composite.CompositeSpatialStrategy
import org.apache.lucene.spatial.prefix.RecursivePrefixTreeStrategy
import org.apache.lucene.spatial.prefix.tree.GeohashPrefixTree
import org.apache.lucene.spatial.query.{SpatialArgs, SpatialOperation}
import org.apache.lucene.spatial.serialized.SerializedDVStrategy
import org.apache.lucene.spatial3d.geom.{GeoPoint, PlanetModel}
import org.apache.lucene.store.{RAMDirectory, SimpleFSDirectory}
import org.junit.runner.RunWith
import org.locationtech.spatial4j.context.SpatialContext
import org.locationtech.spatial4j.distance.DistanceUtils
import org.locationtech.spatial4j.shape.{Point, Shape}
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

case class WorldCity
(
  id: Int,
  country: String,
  city: String,
  accentCity: String,
  region: String,
  population: Option[Long],
  latitude: Double,
  longitude: Double
)

@RunWith(classOf[JUnitRunner])
class GeolocationTest extends FlatSpec with TestCommon {

  val filePath = "/Users/mtovbin/Downloads/world-cities-database/worldcitiespop.csv"
  val citiesCsv = scala.io.Source.fromFile(filePath, "UTF-8").getLines().drop(1)
  var id = -1
  val cities = citiesCsv.map { c =>
    val Array(country, city, accentCity, region, population, latitude, longitude) = c.split(',')
    id = id + 1
    WorldCity(
      id = id,
      country = country,
      city = city,
      accentCity = accentCity,
      region = region,
      population = if (population.isEmpty) None else Some(population.toDouble.toLong),
      latitude = latitude.toDouble,
      longitude = longitude.toDouble
    )
  }.toSeq
  cities.take(1000).foreach(println)


  val ctx = SpatialContext.GEO
  val grid = new GeohashPrefixTree(ctx, 11)
  val fieldName = "myGeoField"
//  val strategy = new CompositeSpatialStrategy(fieldName,
//    new RecursivePrefixTreeStrategy(grid, fieldName),
//    new SerializedDVStrategy(ctx, fieldName)
//  )
  val strategy = new RecursivePrefixTreeStrategy(grid, fieldName)
  val path = Paths.get("/tmp/geo-index")
  val directory = new SimpleFSDirectory(path)// new RAMDirectory()

//  val iwConfig: IndexWriterConfig = new IndexWriterConfig(null)
//  val indexWriter: IndexWriter = new IndexWriter(directory, iwConfig)
//
//  cities.foreach { city =>
//    val point = ctx.getShapeFactory.pointXY(city.longitude, city.latitude) // X - longitude, Y - latitude
//    val document = newSampleDocument(city.id, point)
//    indexWriter.addDocument(document)
//  }
//  indexWriter.close()

  println("Opening index...")
  val indexReader = DirectoryReader.open(directory)
  println("Starting index searcher...")
  val indexSearcher = new IndexSearcher(indexReader)

  val citiesById = cities.groupBy(_.id).mapValues(_.head)


  // Search by radius
  val idSort = new Sort(new SortField("id", SortField.Type.INT))
  val circle = ctx.getShapeFactory.circle(-122.1430, 37.4419, DistanceUtils.dist2Degrees(200, DistanceUtils.EARTH_MEAN_RADIUS_KM))
  val args = new SpatialArgs(SpatialOperation.Intersects, circle)
  val query = strategy.makeQuery(args)
  var total = 0L
  println("Running queries...")
  (1 to 10000).foreach { _ =>
    val start = System.currentTimeMillis()
    val docs: TopDocs = indexSearcher.search(query, 10, idSort)
    val end = System.currentTimeMillis()
    total = total + (end - start)
    println((end - start) + " elapsed ms (by RADIUS)")
    docs.scoreDocs.foreach { hit =>
      println(hit.doc + " -> " + citiesById(hit.doc))
    }
  }
  println( (total / 10000.0) + " elapsed ms AVERAGE -------- (by RADIUS)")


  // Search by distance
  // abs(lat) <= 90, abs(long) <= 180
  val pt = ctx.getShapeFactory.pointXY(-122.1430, 37.4419)
  val valueSource: DoubleValuesSource = strategy.makeDistanceValueSource(pt, DistanceUtils.DEG_TO_KM)
  //the distance (in km)
  val distSort = new Sort(valueSource.getSortField(false)).rewrite(indexSearcher)
  //false=asc dist

  (1 to 10).foreach { _ =>
    val start = System.currentTimeMillis()
    val docs: TopDocs = indexSearcher.search(new MatchAllDocsQuery, 10, distSort)
    val end = System.currentTimeMillis()
    println((end - start) + " elapsed ms (BY DISTANCE)")
    docs.scoreDocs.foreach { hit =>
      println(hit.doc + " -> " + citiesById(hit.doc))
    }
  }



  //Spatial4j is x-y order for arguments
  // indexWriter.addDocument(newSampleDocument(2, ctx.makePoint(-(80.93), 33.77)))
  //Spatial4j has a WKT parser which is also "x y" order
  // indexWriter.addDocument(newSampleDocument(4, ctx.readShapeFromWkt("POINT(60.9289094 -50.7693246)")))
  // indexWriter.addDocument(newSampleDocument(20, ctx.makePoint(0.1, 0.1), ctx.makePoint(0, 0)))


  def newSampleDocument(id: Int, shapes: Shape*): Document = {
    val doc = new Document()
    doc.add(new StoredField("id", id))
    doc.add(new NumericDocValuesField("id", id))
    //Potentially more than one shape in this field is supported by some
    // strategies; see the javadocs of the SpatialStrategy impl to see.
    for (shape <- shapes) {
      for (f <- strategy.createIndexableFields(shape)) {
        doc.add(f)
      }
      //store it too; the format is up to you
      //  (assume point in this example)
      val pt = shape.asInstanceOf[Point]
      doc.add(new StoredField(strategy.getFieldName, pt.getX + " " + pt.getY))
    }
    doc
  }


  val PaloAlto: (Double, Double) = (37.4419, -122.1430)

  Spec[Geolocation] should "extend OPList[Double]" in {
    val myGeolocation = new Geolocation(List.empty[Double])
    myGeolocation shouldBe a[FeatureType]
    myGeolocation shouldBe a[OPCollection]
    myGeolocation shouldBe a[OPList[_]]
  }

  it should "behave on missing data" in {
    val sut = new Geolocation(List.empty[Double])
    sut.lat.isNaN shouldBe true
    sut.lon.isNaN shouldBe true
    sut.accuracy shouldBe GeolocationAccuracy.Unknown
  }

  it should "not accept missing value" in {
    assertThrows[IllegalArgumentException](new Geolocation(List(PaloAlto._1)))
    assertThrows[IllegalArgumentException](new Geolocation(List(PaloAlto._1, PaloAlto._2)))
    assertThrows[IllegalArgumentException](new Geolocation((PaloAlto._1, PaloAlto._2, 123456.0)))
  }

  it should "compare values correctly" in {
    new Geolocation(List(32.399, 154.213, 6.0)).equals(new Geolocation(List(32.399, 154.213, 6.0))) shouldBe true
    new Geolocation(List(12.031, -23.44, 6.0)).equals(new Geolocation(List(32.399, 154.213, 6.0))) shouldBe false
    FeatureTypeDefaults.Geolocation.equals(new Geolocation(List(32.399, 154.213, 6.0))) shouldBe false
    FeatureTypeDefaults.Geolocation.equals(FeatureTypeDefaults.Geolocation) shouldBe true
    FeatureTypeDefaults.Geolocation.equals(Geolocation(List.empty[Double])) shouldBe true

    (35.123, -94.094, 5.0).toGeolocation shouldBe a[Geolocation]
  }

  it should "correctly generate a Lucene GeoPoint object" in {
    val myGeo = new Geolocation(List(32.399, 154.213, 6.0))
    myGeo.toGeoPoint shouldBe new GeoPoint(PlanetModel.WGS84, math.toRadians(myGeo.lat), math.toRadians(myGeo.lon))
  }

}
