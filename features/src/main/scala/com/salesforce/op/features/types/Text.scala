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

import java.io.InputStream
import java.nio.charset.StandardCharsets
import java.util.regex.Pattern

import com.twitter.chill.Base64.{InputStream => Base64InputStream}
import org.apache.commons.httpclient.URI
import org.apache.commons.io.input.CharSequenceInputStream
import org.apache.commons.validator.routines.UrlValidator

/**
 * Text value representation
 *
 * A base class for all the text Feature Types
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

/**
 * Email value representation
 *
 * @param value email value
 */
class Email(value: Option[String]) extends Text(value) {
  def this(value: String) = this(Option(value))
  /**
   * Extract email prefix
   * @return if email is invalid or empty - None is returned; otherwise some value with prefix
   */
  def prefix: Option[String] = Email.prefixOrDomain(this, isPrefix = true)
  /**
   * Extract email domain
   * @return if email is invalid or empty - None is returned; otherwise some value with domain
   */
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
/**
 * Base64 encoded binary value representation
 *
 * @param value base64 encoded binary value
 */
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

/**
 * Phone number value representation, i.e. '+1-650-113-111-2222'
 *
 * @param value phone number
 */
class Phone(value: Option[String]) extends Text(value){
  def this(value: String) = this(Option(value))
}
object Phone {
  def apply(value: Option[String]): Phone = new Phone(value)
  def apply(value: String): Phone = new Phone(value)
  def empty: Phone = FeatureTypeDefaults.Phone
}

/**
 * Unique identifier value representation
 *
 * @param value unique identifier
 */
class ID(value: Option[String]) extends Text(value){
  def this(value: String) = this(Option(value))
}
object ID {
  def apply(value: Option[String]): ID = new ID(value)
  def apply(value: String): ID = new ID(value)
  def empty: ID = FeatureTypeDefaults.ID
}

/**
 * URL value representation
 *
 * @param value url
 */
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
   * Extracts url domain, i.e. 'salesforce.com', 'data.com' etc.
   *
   * @param escaped true if URI character sequence is in escaped form. false otherwise.
   */
  def domain(escaped: Boolean = false): Option[String] = value map (s => new java.net.URL(new URI(s, escaped).toString).getHost)
  /**
   * Extracts url protocol, i.e. http, https, ftp etc.
   *
   * @param escaped true if URI character sequence is in escaped form. false otherwise.
   */
  def protocol(escaped: Boolean = false): Option[String] = value map (s => new java.net.URL(new URI(s, escaped).toString).getProtocol)
}
object URL {
  def apply(value: Option[String]): URL = new URL(value)
  def apply(value: String): URL = new URL(value)
  def empty: URL = FeatureTypeDefaults.URL
}

/**
 * Large text values (more than 4000 bytes)
 *
 * @param value large text value
 */
class TextArea(value: Option[String]) extends Text(value){
  def this(value: String) = this(Option(value))
}
object TextArea {
  def apply(value: Option[String]): TextArea = new TextArea(value)
  def apply(value: String): TextArea = new TextArea(value)
  def empty: TextArea = FeatureTypeDefaults.TextArea
}

/**
 * A single text value that represents a single selection from a set of values
 *
 * @param value selected text
 */
class PickList(value: Option[String]) extends Text(value) with SingleResponse {
  def this(value: String) = this(Option(value))
}
object PickList {
  def apply(value: Option[String]): PickList = new PickList(value)
  def apply(value: String): PickList = new PickList(value)
  def empty: PickList = FeatureTypeDefaults.PickList
}
/**
 * A single text value that represents a selection from a set of values or a user specified one
 *
 * @param value selected or user specified text
 */
class ComboBox(value: Option[String]) extends Text(value){
  def this(value: String) = this(Option(value))
}
object ComboBox {
  def apply(value: Option[String]): ComboBox = new ComboBox(value)
  def apply(value: String): ComboBox = new ComboBox(value)
  def empty: ComboBox = FeatureTypeDefaults.ComboBox
}

/**
 * Country value representation, i.e. 'United States of America', 'France" etc.
 *
 * @param value country
 */
class Country(value: Option[String]) extends Text(value) with Location {
  def this(value: String) = this(Option(value))
}
object Country {
  def apply(value: Option[String]): Country = new Country(value)
  def apply(value: String): Country = new Country(value)
  def empty: Country = FeatureTypeDefaults.Country
}

/**
 * State value representation, i.e. 'CA', 'OR' etc.
 *
 * @param value state
 */
class State(value: Option[String]) extends Text(value) with Location {
  def this(value: String) = this(Option(value))
}
object State {
  def apply(value: Option[String]): State = new State(value)
  def apply(value: String): State = new State(value)
  def empty: State = FeatureTypeDefaults.State
}

/**
 * Postal code value representation, i.e. '92101', '72212-341' etc.
 *
 * @param value postal code
 */
class PostalCode(value: Option[String]) extends Text(value) with Location {
  def this(value: String) = this(Option(value))
}
object PostalCode {
  def apply(value: Option[String]): PostalCode = new PostalCode(value)
  def apply(value: String): PostalCode = new PostalCode(value)
  def empty: PostalCode = FeatureTypeDefaults.PostalCode
}

/**
 * City value representation, i.e. 'New York', 'Paris' etc.
 *
 * @param value city
 */
class City(value: Option[String]) extends Text(value) with Location {
  def this(value: String) = this(Option(value))
}
object City {
  def apply(value: Option[String]): City = new City(value)
  def apply(value: String): City = new City(value)
  def empty: City = FeatureTypeDefaults.City
}

/**
 * Street representation, i.e. '123 University Ave' etc.
 *
 * @param value street
 */
class Street(value: Option[String]) extends Text(value) with Location {
  def this(value: String) = this(Option(value))
}
object Street {
  def apply(value: Option[String]): Street = new Street(value)
  def apply(value: String): Street = new Street(value)
  def empty: Street = FeatureTypeDefaults.Street
}
