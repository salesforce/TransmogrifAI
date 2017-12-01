/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.reflection

import scala.reflect._
import scala.reflect.runtime.universe._
import scala.reflect.runtime.{universe => runtimeUniverse}
import scala.util.Success


/**
 * Various Reflection helpers
 */
object ReflectionUtils {

  /**
   * Default class loader - Thread.currentThread().getContextClassLoader
   */
  def defaultClassLoader: ClassLoader = Thread.currentThread().getContextClassLoader

  /**
   * Get a runtime mirror
   *
   * @param classLoader class loader to use
   * @return runtime mirror
   */
  def runtimeMirror(classLoader: ClassLoader = defaultClassLoader): Mirror = {
    runtimeUniverse.runtimeMirror(classLoader)
  }

  /**
   * Given a Class this will return the runtime mirror and class mirror
   *
   * @param klazz
   * @param classLoader class loader to use
   * @return tuple of runtimemirrir and class mirror
   */
  def mirrors(klazz: Class[_], classLoader: ClassLoader = defaultClassLoader): (Mirror, ClassMirror) = {
    val rtm = runtimeMirror()
    val klazzSymbol = rtm.classSymbol(klazz)
    rtm -> rtm.reflectClass(klazzSymbol)
  }

  /**
   * Find the best constructor method to use.
   *
   * @param klazz
   * @return ctor method with params list
   */
  def bestCtor(klazz: Class[_]): (MethodMirror, List[List[Symbol]]) = {
    val (runtimeMirror, classMirror) = mirrors(klazz)
    val tMembers = runtimeMirror.classSymbol(klazz).toType.members
    bestCtor(classMirror, tMembers)
  }

  /**
   * Create a new instance of T
   *
   * @param klazz
   * @param ctorArgs ctor args getter function
   * @tparam T
   * @return new instance of T
   */
  def newInstance[T](klazz: Class[_], ctorArgs: (String, Symbol) => util.Try[Any]): T = {
    val (ctor, ctorParams) = ReflectionUtils.bestCtor(klazz)
    val args = extractParams(klazz, ctorParams, ctorArgs)
    ctor.apply(args.map(_._2): _*).asInstanceOf[T]
  }

  /**
   * Copy any instance using reflection.
   *
   * Note: all the implicit type values must be explicitly mentioned in the ctor.
   *
   * So instead of implicitly passing the TypeTag:
   * case class MyClass[T : TypeTag](t: T)
   *
   * Explicitly pass TypeTag as a val:
   * case class MyClass[T](t: T)(implicit val tt: TypeTag[T])
   *
   * @param t instance of type T
   * @tparam T type T
   * @return a copy of t
   * @throws RuntimeException in case a ctor param value cannot be extracted
   */
  def copy[T: ClassTag](t: T): T = {
    val (ctor, args) = bestCtorWithArgs(t)
    ctor.apply(args.map(_._2): _*).asInstanceOf[T]
  }

  /**
   * Use reflection to get the primary constructor definition of an instance.
   * Along with the constructor Mirror, this will also search the instance mirror for
   * the appropriate members and their values used as arguments to the
   * constructor.
   *
   * @param t instance to make copy from
   * @tparam T type of instance to copy
   * @return a tuple where the first element is the Constructor Mirror,
   *         and the second element is a List of tuples containing the
   *         member names and their values
   */
  def bestCtorWithArgs[T: ClassTag](t: T): (MethodMirror, List[(String, Any)]) = {
    val klazz = t.getClass
    val (runtimeMirror, classMirror) = mirrors(klazz)
    val tMembers = runtimeMirror.classSymbol(klazz).toType.members
    val (ctorMethod, ctorParamLists) = bestCtor(classMirror, tMembers)

    val getters = tMembers.collect {
      case m: MethodSymbol if m.isGetter && m.isPublic => termNameStr(m.name) -> m
    }.toMap
    val vals = tMembers.collect {
      case v: TermSymbol if v.isVal => termNameStr(v.name) -> v
    }.toMap

    val instanceMirror = runtimeMirror.reflect(t)

    def ctorArgs(paramName: String, param: Symbol): util.Try[Any] = util.Try {
      val getterOpt = getters.get(paramName).map(instanceMirror.reflectMethod)
      if (getterOpt.isDefined) getterOpt.get.apply()
      else instanceMirror.reflectField(vals(paramName)).get
    }

    ctorMethod -> extractParams(klazz, ctorParamLists, ctorArgs)
  }

  /**
   * Find the class by name
   *
   * @param name class
   * @param classLoader
   * @throws ClassNotFoundException
   * @return class object
   */
  def classForName(name: String, classLoader: ClassLoader = defaultClassLoader): Class[_] = {
    classLoader.loadClass(name)
  }

  /**
   * Create a TypeTag for Type
   *
   * @param rtm runtime mirror (default: [[ReflectionUtils.runtimeMirror]]
   * @param tpe type
   * @tparam T type T
   * @return TypeTag[T]
   */
  def typeTagForType[T](rtm: Mirror = runtimeMirror(), tpe: Type): TypeTag[T] = {
    TypeTag(rtm, new api.TypeCreator {
      def apply[U <: api.Universe with Singleton](m: api.Mirror[U]): U#Type =
        if (m eq rtm) tpe.asInstanceOf[U#Type]
        else throw new IllegalArgumentException(s"Type tag defined in $rtm cannot be migrated to other mirrors.")
    })
  }

  /**
   * Return a TypeTag of typeTag[T].tpe.dealias (see [[TypeApi.dealias]])
   *
   * @param ttag TypeTag[T]
   * @tparam T type T
   * @return TypeTag of typeTag[T].tpe.dealias
   */
  def dealiasedTypeTag[T](implicit ttag: TypeTag[T]): TypeTag[T] = typeTagForType[T](tpe = ttag.tpe.dealias)

  /**
   * Creates a Manifest[T] of a TypeTag[T]
   *
   * @tparam T type T with a TypeTag
   * @return Manifest[T]
   */
  def manifestForTypeTag[T: TypeTag]: Manifest[T] = {
    val t = typeTag[T]
    val mirror = t.mirror

    def toManifestRec(t: Type): Manifest[_] = {
      val clazz = ClassTag[T](mirror.runtimeClass(t)).runtimeClass
      if (t.typeArgs.length == 1) {
        val arg = toManifestRec(t.typeArgs.head)
        ManifestFactory.classType(clazz, arg)
      } else if (t.typeArgs.length > 1) {
        val args = t.typeArgs.map(x => toManifestRec(x))
        ManifestFactory.classType(clazz, args.head, args.tail: _*)
      } else {
        ManifestFactory.classType(clazz)
      }
    }

    toManifestRec(t.tpe).asInstanceOf[Manifest[T]]
  }

  /**
   * Cycles through the parameter list and uses the paramGetter lambda function to extract
   * the param value by name. If a param extraction has failed a RuntimeException will be
   * thrown
   *
   * @param klazz
   * @param paramLists  param list to try and get values for
   * @param paramGetter lambda function to extract the param value by name.
   * @return List of tuples with the param name and it's value
   */
  private def extractParams
  (
    klazz: Class[_],
    paramLists: List[List[Symbol]],
    paramGetter: (String, Symbol) => util.Try[Any]
  ): List[(String, Any)] = {
    val paramValues = for {
      paramList <- paramLists
      param <- paramList
      paramName = termNameStr(param.name.toTermName)
    } yield paramName -> paramGetter(paramName, param)

    val maybeFailure = paramValues.collectFirst {
      case (paramName, maybeValue) if maybeValue.isFailure => (paramName, maybeValue.failed.get)
    }

    if (maybeFailure.isDefined) {
      val ex = maybeFailure.get._2
      throw new RuntimeException(
        s"Failed to extract value for param '${maybeFailure.get._1}' " +
          s"for an instance of type '${klazz.getCanonicalName}' due to: ${ex.getMessage}",
        ex
      )
    }
    paramValues.collect { case (n, Success(v)) => n -> v }
  }

  private def bestCtor(classMirror: ClassMirror, tMembers: MemberScope): (MethodMirror, List[List[Symbol]]) = {
    // TODO: implement a more robust logic of picking up the best ctor
    val ctor = tMembers.collect {
      case m: MethodSymbol if m.isConstructor && m.isPrimaryConstructor && m.isPublic => m
    }.head

    val cMethod = classMirror.reflectConstructor(ctor)
    cMethod -> ctor.paramLists
  }

  private def termNameStr(termName: TermName): String = termName.decodedName.toString.trim

}
