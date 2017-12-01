/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types

import language.postfixOps
import org.apache.lucene.geo.GeoUtils
import enumeratum.values.{IntEnum, IntEnumEntry}
import org.apache.lucene.spatial3d.geom.{GeoPoint, PlanetModel}
import Geolocation._

/**
 * Represented as a list of latitude, longitude, accuracy (only populated if all are present)
 *
 * @param value a list of latitude, longitude, accuracy
 */
class Geolocation(val value: Seq[Double]) extends OPList[Double] with Location {
  validate() // validate the coordinates on construction
  def this(lat: Double, lon: Double, accuracy: GeolocationAccuracy) =
    this(geolocationData(lat, lon, accuracy))

  def this(v: (Double, Double, Double)) = this(geolocationData(v._1, v._2, v._3))
  def lat: Double = if (isEmpty) Double.NaN else value(0)
  def lon: Double = if (isEmpty) Double.NaN else value(1)
  def latitude: Double = lat
  def longitude: Double = lon

  def accuracy: GeolocationAccuracy =
    if (isEmpty) GeolocationAccuracy.Unknown else GeolocationAccuracy.withValue(value(2).toInt)

  def toGeoPoint: GeoPoint = {
    // If this Geolocation object is empty, then return the zero vector as the GeoPoint since we use
    // GeoPoint coordinates in aggregation functions
    if (isEmpty) Geolocation.EmptyGeoPoint
    else new GeoPoint(PlanetModel.WGS84, math.toRadians(lat), math.toRadians(lon))
  }

  /**
   * Validates the coordinates
   * @throws IllegalArgumentException when data are wrong
   */
  def validate(): Unit = {
    require(isEmpty || value.length == 3,
      s"Geolocation must have lat, lon, and accuracy, or be empty: $value")
    if (!isEmpty) Geolocation.validate(lat, lon)
  }

  override def toString: String = {
    if (isEmpty) "Geolocation()" else f"Geolocation($lat%.5f, $lon%.5f, $accuracy)"
  }

}
/**
 * Represented as a list of latitude, longitude, accuracy (only populated if all are present)
 */
object Geolocation {
  private[types] def geolocationData(
    lat: Double,
    lon: Double,
    accuracy: GeolocationAccuracy): Seq[Double] =
    geolocationData(lat, lon, accuracy.value)

  private[types] def geolocationData(
    lat: Double,
    lon: Double,
    accuracy: Double): Seq[Double] = {
    if (lat.isNaN || lon.isNaN) List[Double]() else List(lat, lon, accuracy)
  }

  def apply(lat: Double, lon: Double, accuracy: GeolocationAccuracy): Geolocation =
    new Geolocation(lat = lat, lon = lon, accuracy = accuracy)
  def apply(v: Seq[Double]): Geolocation = new Geolocation(v)
  def apply(v: (Double, Double, Double)): Geolocation = new Geolocation(v)
  def empty: Geolocation = FeatureTypeDefaults.Geolocation
  private val EmptyGeoPoint = new GeoPoint(0.0, 0.0, 0.0)

  def validate(lat: Double, lon: Double): Unit = {
    GeoUtils.checkLatitude(lat)
    GeoUtils.checkLongitude(lon)
  }

  val EquatorInMiles = 24901.0
  val EarthRadius = 3959.0
}

/**
 * Geolocation Accuracy tells you more about the location at the latitude and longitude for a give address.
 * For example, 'Zip' means the latitude and longitude point to the center of the zip code area.
 */
sealed abstract class GeolocationAccuracy
(
  val value: Int,
  val name: String,
  val rangeInMiles: Double) extends IntEnumEntry {
  lazy val rangeInUnits: Double = rangeInMiles / EarthRadius
}

case object GeolocationAccuracy extends IntEnum[GeolocationAccuracy] {
  val values: List[GeolocationAccuracy] = findValues.toList sortBy(_.rangeInMiles)

  def geoUnitsToMiles(u: Double): Double = u * EarthRadius

  // No match for the address was found
  case object Unknown extends GeolocationAccuracy(0, name = "Unknown", rangeInMiles = EquatorInMiles / 2)
  // In the same building
  case object Address extends GeolocationAccuracy(1, name = "Address", rangeInMiles = 0.005)
  // Near the address
  case object NearAddress extends GeolocationAccuracy(2, name = "NearAddress", rangeInMiles = 0.02)
  // Midway point of the block
  case object Block extends GeolocationAccuracy(3, name = "Block", rangeInMiles = 0.05)
  // Midway point of the street
  case object Street extends GeolocationAccuracy(4, name = "Street", rangeInMiles = 0.15)
  // Center of the extended zip code area
  case object ExtendedZip extends GeolocationAccuracy(5, name = "ExtendedZip", rangeInMiles = 0.4)
  // Center of the zip code area
  case object Zip extends GeolocationAccuracy(6, name = "Zip", rangeInMiles = 1.2)
  // Center of the neighborhood
  case object Neighborhood extends GeolocationAccuracy(7, name = "Neighborhood", rangeInMiles = 3.0)
  // Center of the city
  case object City extends GeolocationAccuracy(8, name = "City", rangeInMiles = 12.0)
  // Center of the county
  case object County extends GeolocationAccuracy(9, name = "County", rangeInMiles = 40.0)
  // Center of the state
  case object State extends GeolocationAccuracy(10, name = "State", rangeInMiles = 150.0)

  def forRangeInMiles(miles: Double): GeolocationAccuracy = {
    val result = values.dropWhile(_.rangeInMiles < miles * 0.99).headOption getOrElse Unknown
    result
  }

  def forRangeInUnits(units: Double): GeolocationAccuracy =
    forRangeInMiles(geoUnitsToMiles(units))

  def worst(accuracies: GeolocationAccuracy*): GeolocationAccuracy = {
    forRangeInMiles((Unknown :: accuracies.toList) map (_.rangeInMiles) max)
  }
}
