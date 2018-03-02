/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types

import java.io.InputStream
import java.nio.charset.StandardCharsets
import java.util.regex.Pattern

import com.twitter.chill.Base64.{InputStream => Base64InputStream}
import org.apache.commons.io.input.CharSequenceInputStream
import org.apache.commons.validator.routines.UrlValidator

/**
 * A base class for all the text feature types
 *
 * @param value text value
 */
class Text(val value: Option[String]) extends FeatureType {
  type Value = Option[String]
  def this(value: String) = this(Option(value))
  final def isEmpty: Boolean = value.isEmpty
  final def map[B](f: String => B): Option[B] = value.map(f)
}
object Text {
  def apply(value: Option[String]): Text = new Text(value)
  def apply(value: String): Text = new Text(value)
  def empty: Text = FeatureTypeDefaults.Text
}

class Email(value: Option[String]) extends Text(value) {
  def this(value: String) = this(Option(value))
  def prefix: Option[String] = Email.prefixOrDomain(this, isPrefix = true)
  def domain: Option[String] = Email.prefixOrDomain(this, isPrefix = false)
}
object Email {
  def apply(value: Option[String]): Email = new Email(value)
  def apply(value: String): Email = new Email(value)
  def empty: Email = FeatureTypeDefaults.Email

  // TODO: ideally we'd return an object containing all the parsed email parts (rather than Text), but that
  // depends on how we end up structuring our types, so this can be revisited then.
  // TODO: we should likely be validating email addresses with a more robust library.
  // scalastyle:off
  private val pattern = Pattern.compile(
    """^([a-zA-Z0-9\.!#$%&'*+/=?^_`{|}~-]+)@([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*)$"""
  )
  // scalastyle:on
  private def prefixOrDomain(email: Email, isPrefix: Boolean): Option[String] =
    email.v.map(Email.pattern.matcher).flatMap(m =>
      if (!m.matches()) None else if (isPrefix) Option(m.group(1)) else Option(m.group(2))
    )
}

class Base64(value: Option[String]) extends Text(value) {
  def this(value: String) = this(Option(value))
  /**
   * Input stream over the contents in this base64
   * @return Some(inputStream) if data present, or None if not
   */
  def asInputStream: Option[InputStream] = {
    value map { v => new Base64InputStream(new CharSequenceInputStream(v, StandardCharsets.ISO_8859_1)) }
  }
  /**
   * Maps f over the input stream of the contents this base64
   * @param f function to apply over stream
   * @return Some(inputStream) if data present, or None if not
   * @throws IOException if an I/O error occurs.
   */
  def mapInputStream[T](f: InputStream => T): Option[T] = asInputStream.map(in => try f(in) finally in.close())
  /**
   * Bytes hidden in this base64
   * @return Some(bytes) if data present, or None if not
   */
  def asBytes: Option[Array[Byte]] = value map java.util.Base64.getDecoder.decode
  /**
   * String hidden in this base64
   * @return Some(string) if data present, or None if not
   */
  def asString: Option[String] = asBytes map (new String(_))
}

object Base64 {
  def apply(value: Option[String]): Base64 = new Base64(value)
  def apply(value: String): Base64 = new Base64(value)
  def empty: Base64 = FeatureTypeDefaults.Base64
}

class Phone(value: Option[String]) extends Text(value){
  def this(value: String) = this(Option(value))
}
object Phone {
  def apply(value: Option[String]): Phone = new Phone(value)
  def apply(value: String): Phone = new Phone(value)
  def empty: Phone = FeatureTypeDefaults.Phone
}

class ID(value: Option[String]) extends Text(value){
  def this(value: String) = this(Option(value))
}
object ID {
  def apply(value: Option[String]): ID = new ID(value)
  def apply(value: String): ID = new ID(value)
  def empty: ID = FeatureTypeDefaults.ID
}

class URL(value: Option[String]) extends Text(value){
  def this(value: String) = this(Option(value))
  /**
   * Verifies if the url is of correct form of "Uniform Resource Identifiers (URI): Generic Syntax"
   * RFC2396 (http://www.ietf.org/rfc/rfc2396.txt)
   * Default valid protocols are: http, https, ftp.
   */
  def isValid: Boolean = value.exists(UrlValidator.getInstance().isValid)
  /**
   * Verifies if the url is of correct form of "Uniform Resource Identifiers (URI): Generic Syntax"
   * RFC2396 (http://www.ietf.org/rfc/rfc2396.txt)
   * @param protocols url protocols to consider valid, i.e. http, https, ftp etc.
   */
  def isValid(protocols: Array[String]): Boolean = value.exists(new UrlValidator(protocols).isValid)
  /**
   * Extracts url domain, i.e. salesforce.com, data.com etc.
   */
  def domain: Option[String] = value map (new java.net.URL(_).getHost)
  /**
   * Extracts url protocol, i.e. http, https, ftp etc.
   */
  def protocol: Option[String] = value map (new java.net.URL(_).getProtocol)
}
object URL {
  def apply(value: Option[String]): URL = new URL(value)
  def apply(value: String): URL = new URL(value)
  def empty: URL = FeatureTypeDefaults.URL
}

class TextArea(value: Option[String]) extends Text(value){
  def this(value: String) = this(Option(value))
}
object TextArea {
  def apply(value: Option[String]): TextArea = new TextArea(value)
  def apply(value: String): TextArea = new TextArea(value)
  def empty: TextArea = FeatureTypeDefaults.TextArea
}

class PickList(value: Option[String]) extends Text(value) with SingleResponse {
  def this(value: String) = this(Option(value))
}
object PickList {
  def apply(value: Option[String]): PickList = new PickList(value)
  def apply(value: String): PickList = new PickList(value)
  def empty: PickList = FeatureTypeDefaults.PickList
}

class ComboBox(value: Option[String]) extends Text(value){
  def this(value: String) = this(Option(value))
}
object ComboBox {
  def apply(value: Option[String]): ComboBox = new ComboBox(value)
  def apply(value: String): ComboBox = new ComboBox(value)
  def empty: ComboBox = FeatureTypeDefaults.ComboBox
}

class Country(value: Option[String]) extends Text(value) with Location {
  def this(value: String) = this(Option(value))
}
object Country {
  def apply(value: Option[String]): Country = new Country(value)
  def apply(value: String): Country = new Country(value)
  def empty: Country = FeatureTypeDefaults.Country
}

class State(value: Option[String]) extends Text(value) with Location {
  def this(value: String) = this(Option(value))
}
object State {
  def apply(value: Option[String]): State = new State(value)
  def apply(value: String): State = new State(value)
  def empty: State = FeatureTypeDefaults.State
}

class PostalCode(value: Option[String]) extends Text(value) with Location {
  def this(value: String) = this(Option(value))
}
object PostalCode {
  def apply(value: Option[String]): PostalCode = new PostalCode(value)
  def apply(value: String): PostalCode = new PostalCode(value)
  def empty: PostalCode = FeatureTypeDefaults.PostalCode
}

class City(value: Option[String]) extends Text(value) with Location {
  def this(value: String) = this(Option(value))
}
object City {
  def apply(value: Option[String]): City = new City(value)
  def apply(value: String): City = new City(value)
  def empty: City = FeatureTypeDefaults.City
}

class Street(value: Option[String]) extends Text(value) with Location {
  def this(value: String) = this(Option(value))
}
object Street {
  def apply(value: Option[String]): Street = new Street(value)
  def apply(value: String): Street = new Street(value)
  def empty: Street = FeatureTypeDefaults.Street
}
