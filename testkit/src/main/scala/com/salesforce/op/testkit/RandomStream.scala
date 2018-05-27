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

import scala.annotation.tailrec
import scala.util.Random

/**
 * Random stream of data of given type T
 *
 * @param producer a function that, given an RND, produces one value of type T
 * @tparam T the type of result
 */
case class RandomStream[+T](producer: Random => T)
  extends (Random => InfiniteStream[T]) with Serializable { self =>

  /**
   * Builds an infinite stream, given an rng
   * @param rng a random numbers generator
   * @return a stream of data produced by producer
   */
  def apply(rng: Random): InfiniteStream[T] = new InfiniteStream[T] {
    def next: T = producer(rng)
  }

  /**
   * Applies a function to the results of this stream, streaming the results
   * @param g the function to apply
   * @tparam U the function's result type
   * @return a new stream of values ot type U
   */
  def map[U](g: T => U): RandomStream[U] = new RandomStream[U](g compose producer) {
    override def reset(seed: Long): Unit = {
      self.reset(seed)
    }
  }

  /**
   * Zip with another random stream, producing pairs of values
   * @param anotherStream another stream
   * @tparam U type of values produced by another stream
   * @return as stream of values of type (T, U)
   */
  def zipWith[U](anotherStream: RandomStream[U]): RandomStream[(T, U)] =
    new RandomStream[(T, U)](rng => (producer(rng), anotherStream.producer(rng))) {
      override def reset(seed: Long): Unit = {
        self.reset(seed)
        anotherStream.reset(seed)
      }
    }

  /**
   * Default reset - does nothing; probably it's overridden in subclasses
   * @param seed the seed for reset
   */
  def reset(seed: Long): Unit = ()
}

/**
 * Random streams factory: bits, longs, doubles etc.
 */
object RandomStream {

  /**
   * Producer of true with a given probability
   * @param probabilityOfTrue the probability of true
   * @return the function that takes a random and returns a boolean
   */
  def trueWithProbability(probabilityOfTrue: Double): Random => Boolean =
    (rng: Random) => rng.nextDouble <= probabilityOfTrue

  /**
   * Random stream of bits (booleans)
   * The stream should be passed an RNG to produce values
   *
   * @param probabilityOfTrue the probability of the boolean being true
   * @return the random stream of booleans with given probability of truths
   */
  def ofBits(probabilityOfTrue: Double): RandomStream[Boolean] =
    RandomStream(trueWithProbability(probabilityOfTrue))

  /**
   * Random stream of 1s and 0s
   * The stream should be passed an RNG to produce values
   *
   * @param probabilityOfTrue the probability of the boolean being true
   * @return the random stream of zeros and ones with given probability of ones
   */
  def ofOnesAndZeros(probabilityOfTrue: Double): RandomStream[Double] =
    ofBits(probabilityOfTrue) map (b => if (b) 1.0 else 0.0)

  /**
   * Random stream of some bits (booleans)
   * The stream should be passed an RNG to produce values
   *
   * @param probabilityOfTrue the probability of the boolean being true
   * @return the random stream of options of booleans with given probability of truths
   */
  def ofBitOptions(probabilityOfTrue: Double): RandomStream[Option[Boolean]] =
    RandomStream(rng => Option(rng.nextDouble <= probabilityOfTrue))

  /**
   * Produces a stream of bits (booleans), with the given probability of true values
   *
   * @param seed              the seed for random numbers generator
   * @param probabilityOfTrue probability of value being true
   * @return the infinite stream of booleans
   */
  def ofBits(seed: Long, probabilityOfTrue: Double): InfiniteStream[Boolean] =
    ofBits(probabilityOfTrue)(new Random(seed))

  /**
   * Random stream of longs
   * The stream should be passed an RNG to produce values
   *
   * @return the random stream of longs
   */
  def ofLongs: RandomStream[Long] = RandomStream(_.nextLong)

  /**
   * Random stream of longs
   * The stream should be passed an RNG to produce values in a given range
   *
   * @param from minimum value to produce (inclusive)
   * @param to   maximum value to produce (exclusive)
   *
   * @return the random stream of longs between from and to
   */
  def ofLongs(from: Long, to: Long): RandomStream[Long] =
    RandomStream(rng => trim(rng.nextLong, from, to))

  private def trim(value: Long, from: Long, to: Long) = {
    val d = to - from
    val candidate = value % d
    (candidate + d) % d + from
  }

  /**
   * An incrementing stream of random longs
   * @param init initial value
   * @param minDelta minimum step to the next
   * @param maxDelta maximum step to the next; if it is the same as minDelta, values are not random
   * @return a random stream of longs
   */
  def incrementing(init: Long, minDelta: Long, maxDelta: Long): RandomStream[Long] = {
    val range = maxDelta + 1 - minDelta
    var s: Long = init

    new RandomStream(rnd => {
      val d = if (range == 0) minDelta else minDelta + (rnd.nextLong % range + range) % range
      s += d
      s
    }) {
      override def reset(seed: Long): Unit = {
        s = init
      }
    }
  }

  /**
   * Generator of values from a given collection,
   * by default uniformly distributed - or with the distribution provided
   *
   * @param elements all possible elements this generator can produce
   * @tparam T type of the values
   */
  case class CollectionGenerator[T](elements: Seq[T]) extends (Random => T) with Serializable {

    /**
     * produces one value from the given collection
     *
     * @param rng random number generator used
     * @return a random value, distributed uniformly by default, or with
     *         the given distribution law
     */
    def apply(rng: Random): T = elements(rng.nextInt(elements.length))

    /**
     * Provides the distribution law with which to return values of the collection
     *
     * @param dist cumulative distribution function. The sequence should be
     *             of the same length as the collection. The value at position k
     *             is the probability to hit an element in the collection between 0 and k.
     * @return a new collection generator, that generates values at this given distribution
     */
    def distributedAs(dist: Seq[Double]): CollectionGenerator[T] =
      dist.size match {
        case 0 => this
        case n if n == elements.size => new CollectionGenerator[T](elements) {
          override def apply(rng: Random): T = {
            elements(binarySearch(dist, rng.nextDouble))
          }
        }
        case n => throw new IllegalArgumentException(
          s"Distribution data size $n, must be ${elements.size}")
      }


    // random doubles are generated by RealNumbersGenerator, with a variety of distributions
    // they use a different rng source (from Spark), so we can't use Random for it
    private def binarySearch(data: Seq[Double], target: Double): Int = {
      @tailrec
      def bsf(start: Int, end: Int): Int = {
        if (start > end) start
        else {
          val mid = start + (end - start + 1) / 2
          data(mid) compareTo target match {
            case -1 => bsf(mid + 1, end)
            case 0 => mid
            case 1 => bsf(start, mid - 1)
          }
        }
      }

      bsf(0, data.length - 1)
    }
  }

  /**
   * Builds a collection generator for a given collection
   *
   * @param elements collection of the values that the generator will produce
   * @tparam T type of data
   * @return a collection generator, uniform distribution by default
   */
  def of[T](elements: Seq[T]): CollectionGenerator[T] = new CollectionGenerator[T](elements)

  /**
   * a generator of random integers between two given numbers (in
   *
   * @param from minimium number to produce (inclusive)
   * @param to   max number to produce (exclusive)
   * @return
   */
  def randomBetween(from: Int, to: Int): Random => Int = {
    val min = math.max(0, from)
    val max = math.max(1, math.max(from, to))
    rng => if (min == max) min else min + rng.nextInt(max - min)
  }

  /**
   * A producer of chunks of data of given type
   *
   * @param min  minimum number of values in the chunk (inclusive)
   * @param max  maximum number of values in the chunk (exclusive)
   * @param source the random source of single values of type T
   * @tparam T data type
   * @return a generator producing random chunks of random data
   */
  private[testkit] def randomChunks[T](min: Int, max: Int = -1)(source: Random => T):
  Random => Seq[T] = rng => {
    val range = 0 until randomBetween(min, max)(rng)
    range map (_ => source(rng))
  }

  /**
   * Builds a generator of chunks of data of given type
   *
   * @param minLen  minimum number of values in the chunk (inclusive)
   * @param maxLen  maximum number of values in the chunk (exclusive); same as minLen if -1
   * @param source the random source of single values of type T
   * @tparam T data type
   * @return a generator producing random chunks of random data
   */
  private[testkit] def ofChunks[T](minLen: Int, maxLen: Int = -1)(source: Random => T):
    RandomStream[Seq[T]] =
    RandomStream(randomChunks[T](minLen, maxLen)(source))


  private[testkit] def groupInChunks[T](min: Int, max: Int = -1)(singles: RandomStream[T]):
  RandomStream[Seq[T]] = new RandomStream[Seq[T]](rng => {
    val range = 0 until randomBetween(min, max)(rng)
    range map (_ => singles.producer(rng))
  })
}
