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

package com.salesforce.op.testkit

import java.net.URLEncoder

import com.salesforce.op.features.types._
import com.salesforce.op.testkit.DataSources._
import com.salesforce.op.testkit.RandomStream._

import scala.language.postfixOps
import scala.reflect.runtime.universe.WeakTypeTag
import scala.util.Random

/**
 * Generator of random text-related FeatureTypes
 *
 * @param stream           produces a stream of random strings
 * @tparam DataType the type of generated results, a FeatureType that is a subtype of Text
 */
case class RandomText[DataType <: Text : WeakTypeTag](stream: RandomStream[String])
  extends StandardRandomData[DataType](sourceOfData = stream map (Option(_)))
    with ProbabilityOfEmpty

/**
 * Generators of all kinds of random text data
 */
object RandomText {

  /**
   * Generator of random text-related feature types
   * @param producer produces random strings
   * @tparam DataType the type of data we produce
   * @return an instance of RandomText
   */
  def apply[DataType <: Text : WeakTypeTag](producer: Random => String): RandomText[DataType] =
    RandomText[DataType](new RandomStream[String](producer))

  /**
   * Builds a factory of generators of random strings of random sizes
   *
   * @param minLen minimum length of random string (inclusive)
   * @param maxLen maximum length of random string (exclusive);
   *               if maxLen = -1, strings are of the same length
   * @tparam T the type of data this generator generates
   * @return a random text generator for the given type
   */
  private def justText[T <: Text : WeakTypeTag](minLen: Int, maxLen: Int): RandomText[T] = {
    val length = randomBetween(minLen, maxLen)
    RandomText[T]((rng: Random) => rng.nextString(length(rng)))
  }

  /**
   * strings(min, max) produces strings of specified length(s)
   *
   * @param minLen minimum length of a random string (inclusive)
   * @param maxLen maximum length of a random string (exclusive)
   * @return a random text generator
   */
  def strings(minLen: Int, maxLen: Int): RandomText[Text] = justText[Text](minLen, maxLen)

  /**
   * textAreas(min, max) produces textareas of specified length(s)
   *
   * @param minLen minimum length of a random string (inclusive)
   * @param maxLen maximum length of a random string (exclusive)
   * @return a random textarea generator
   */
  def textAreas(minLen: Int, maxLen: Int): RandomText[TextArea] = justText[TextArea](minLen, maxLen)

  private val anEmailId = (rng: Random) => {
    val first = RandomFirstName(rng)
    val last = RandomLastName(rng)
    if (rng.nextBoolean) s"$first.$last" else {
      val plainId = s"${first.toLowerCase.head}${last.toLowerCase}"
      if (rng.nextBoolean) plainId else plainId + rng.nextInt(999)
    }
  }

  /**
   * Produces emails in the specified domain
   *
   * @param domain email domain (e.g. in "petrov@mid.ru": "mid.ru" is domain)
   * @return a random emails generator
   */
  def emails(domain: String): RandomText[Email] = {
    val producer = (rng: Random) => anEmailId(rng) + "@" + domain

    RandomText[Email](producer)
  }

  /**
   * Produces emails in the specified collection of random domains
   * domains can be provided with probabilities, e.g.
   * {{{
   * emailsOn(RandomStream of List("gitmo.mil", "kremlin.ru"))
   * emailsOn(RandomStream of List("gitmo.mil", "kremlin.ru") distributedAs List(0.9, 1.0))
   * }}}
   *
   * @param domains producer of random email domains
   *                (e.g. in "petrov@mid.ru": "mid.ru" is domain)
   * @return a random emails generator
   */
  def emailsOn(domains: Random => String): RandomText[Email] = {
    val namesWithDomains = new RandomStream(anEmailId) zipWith RandomStream(domains)
    val streamOfEmails = namesWithDomains map { case(name, domain) => s"$name@$domain" }

    RandomText[Email](streamOfEmails)
  }

  /**
   * Produces random elements (strings) from a list, with the provided distribution
   *
   * @param domain       list of possible values returned by this generator
   * @param distribution distribution of probability for this list - if provided;
   *                     if not, distribution is assumed to be uniform.
   *                     distribution must be an array of double values, same size
   *                     as domain; distribution(k) is the probability that
   *                     an element in domain
   * @return a random text generator that only returns elements of domain
   */
  private def selectRandom[T <: Text : WeakTypeTag]
  (
    domain: List[String], distribution: Seq[Double] = Nil
  ): RandomText[T] = RandomText[T](RandomStream of domain distributedAs distribution)

  /**
   * Produces random Text from a given list of possible values
   *
   * @param domain       list of possible values returned by this generator
   * @param distribution distribution of probability for this list - if provided;
   *                     if not, distribution is assumed to be uniform.
   *                     distribution must be an array of double values, same size
   *                     as domain; distribution(k) is the probability that
   *                     an element in domain
   * @return a random picklist generator that only returns elements of domain
   */
  def textFromDomain(domain: List[String], distribution: Seq[Double] = Nil): RandomText[Text] =
    selectRandom[Text](domain, distribution)

  /**
   * Produces random TextArea from a given list of possible values
   *
   * @param domain       list of possible values returned by this generator
   * @param distribution distribution of probability for this list - if provided;
   *                     if not, distribution is assumed to be uniform.
   *                     distribution must be an array of double values, same size
   *                     as domain; distribution(k) is the probability that
   *                     an element in domain
   * @return a random picklist generator that only returns elements of domain
   */
  def textAreaFromDomain(domain: List[String], distribution: Seq[Double] = Nil): RandomText[TextArea] =
    selectRandom[TextArea](domain, distribution)

  /**
   * Produces random picklists from a given list of possible values
   *
   * @param domain       list of possible values returned by this generator
   * @param distribution distribution of probability for this list - if provided;
   *                     if not, distribution is assumed to be uniform.
   *                     distribution must be an array of double values, same size
   *                     as domain; distribution(k) is the probability that
   *                     an element in domain
   * @return a random picklist generator that only returns elements of domain
   */
  def pickLists(domain: List[String], distribution: Seq[Double] = Nil): RandomText[PickList] =
    selectRandom[PickList](domain, distribution)

  /**
   * Produces random comboboxes from a given list of possible values
   *
   * @param domain       list of possible values returned by this generator
   * @param distribution distribution of probability for this list - if provided;
   *                     if not, distribution is assumed to be uniform.
   *                     distribution must be an array of double values, same size
   *                     as domain; distribution(k) is the probability that
   *                     an element in domain
   * @return a random comboboxes generator that only returns elements of domain
   */
  def comboBoxes(domain: List[String], distribution: Seq[Double] = Nil): RandomText[ComboBox] =
    selectRandom[ComboBox](domain, distribution)

  /**
   * Produces random comboboxes from random Strings
   *
   * @return a random comboboxes generator
   */
  val randomComboBoxes: RandomText[ComboBox] = justText[ComboBox](1, 20)

  /**
   * Produces random country names from a list that we keep in our resource file
   *
   * @return a random country names generator
   */
  val countries: RandomText[Country] = selectRandom[Country](Countries)

  /**
   * Produces random US state names from a list that we keep in our resource file
   *
   * @return a random state names generator
   */
  val states: RandomText[State] = selectRandom[State](States)

  /**
   * Produces random California city names from a list that we keep in our resource file
   *
   * @return a random city names generator
   */
  val cities: RandomText[City] = selectRandom(CitiesOfCalifornia)

  /**
   * Produces random San Jose street names from a list that we keep in our resource file
   *
   * @return a random street names generator
   */
  val streets: RandomText[Street] = selectRandom[Street](StreetsOfSanJose)

  /**
   * Produces a random Base64 strings generator
   *
   * @param minLen minimum source string length
   * @param maxLen maximum source string length
   * @return a generator that returns random well-formed base64 strings
   */
  def base64(minLen: Int, maxLen: Int): RandomText[Base64] = {
    val lengths = randomBetween(minLen, maxLen + 1)
    val b64 = java.util.Base64.getEncoder

    def randomBytes(rng: Random): Array[Byte] = {
      val buf = new Array[Byte](lengths(rng))
      rng.nextBytes(buf)
      buf
    }

    RandomText[Base64]((rng: Random) => new String(b64.encode(randomBytes(rng))))
  }

  private val goodUsPhones = (rng: Random) =>
    RandomAreaCode(rng) + (5550000 + rng.nextInt(10000))

  private val badUsPhones = (rng: Random) => {
    val raw = 10000000000L + (rng.nextInt(199) * 10000000L + rng.nextInt(10000000)) toString

    raw.substring(rng.nextInt(2), 6 + rng.nextInt(raw.length - 6))
  }

  /**
   * A generator of random US phone numbers
   */
  val phones: RandomText[Phone] = RandomText[Phone](goodUsPhones)

  /**
   * A generator of random US phone numbers with some errors
   */
  def phonesWithErrors(probabilityOfError: Double): RandomText[Phone] = {
    val bad = trueWithProbability(probabilityOfError)
    val producer = (rng: Random) => if (bad(rng)) badUsPhones(rng) else goodUsPhones(rng)

    RandomText[Phone](producer)
  }

  /**
   * A generator of random US postal codes
   */
  def postalCodes: RandomText[PostalCode] = {
    def producer = (rng: Random) => (101000 + rng.nextInt(99000)).toString tail

    RandomText[PostalCode](producer)
  }

  /**
   * A generator of random IDs, length from 1 to 41 char. An id consists of alphanumerics and '_'
   */
  val ids: RandomText[ID] = {
    val charsOfID = "_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    RandomText[ID](randomStringOfAlphabet(charsOfID, 1, 41))
  }

  /**
   * A generator of random unique IDs, length at least 3, no upper limit (they are unique!).
   * A unique id consists of alphanumerics and '_', followed by an '_' and a unique serial number
   */
  val uniqueIds: RandomText[ID] = {
    val charsOfID = "_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    val randoms = new RandomStream(randomStringOfAlphabet(charsOfID, 1, 20))
    val numbers = RandomStream.incrementing(1, 1, 1)
    val ids = randoms zipWith numbers map { case(id0, n) => id0 + "_" + n.toHexString }
    RandomText[ID](ids)
  }

  /**
   * A generator of random URLs
   */
  val urls: RandomText[URL] = {
    val lowerCaseAlphaNum = "abcdefghijklmnopqrstuvwxyz0123456789"
    val urlChars = lowerCaseAlphaNum + "-"

    def label(rng: Random) = {
      randomStringOfAlphabet(lowerCaseAlphaNum, 1, 2)(rng) +
        randomStringOfAlphabet(urlChars, 1, 3)(rng) +
        randomStringOfAlphabet(lowerCaseAlphaNum, 1, 2)(rng)
    }

    val domainName = (rng: Random) => {
      1 to (1 + rng.nextInt(3)) map (_ => label(rng)) mkString "."
    }

    urlsOn(domainName)
  }

  /**
   * A generator of random URLs on a given source of random domains
   * domains can be provided with probabilities, e.g.
   * {{{
   * emails(RandomStream of List("gitmo.mil", "kremlin.ru"))
   * emails(RandomStream of List("gitmo.mil", "kremlin.ru") distributedAs List(0.9, 1.0))
   * }}}
   *
   * @param domains producer of random email domains
   *                (e.g. in "petrov@mid.ru": "mid.ru" is domain)
   * @return a producer of random urls
   */
  def urlsOn(domains: Random => String): RandomText[URL] = {
    val protocols = RandomStream.of("http" :: "https" :: Nil)

    def params(rng: Random): String = {
      0 to rng.nextInt(3) map { _ =>
        RandomFirstName(rng).take(3) + "=" + URLEncoder.encode(rng.nextString(2), "UTF-8")
      } mkString "&"
    }

    val producer = (rng: Random) => {
      val dn = domains(rng)
      val par = params(rng)
      val addr = if (par.isEmpty) dn else s"$dn?$par"
      new java.net.URL(protocols(rng) + "://" + addr).toString
    }

    RandomText[URL](producer)
  }

  /**
   * Builds a producer of random strings in a given alphabet
   *
   * @param alphabet characters that are allowed to use in strings
   * @param minLen   minimum length of generated string
   * @param maxLen   maximum length of generated string
   * @return a function that takes an rng and produces a "random" string from the alphabet
   */
  private def randomStringOfAlphabet(alphabet: String, minLen: Int, maxLen: Int): Random => String = {
    val chars = RandomStream.of(alphabet)
    (rng: Random) => RandomStream.randomChunks(minLen, maxLen)(chars)(rng) mkString ""
  }
}
