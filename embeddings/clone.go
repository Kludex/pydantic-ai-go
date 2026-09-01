package embeddings

import "reflect"

func cloneMap(value map[string]any) map[string]any {
	if value == nil {
		return nil
	}
	cloned := make(map[string]any, len(value))
	for key, item := range value {
		cloned[key] = cloneCollection(reflect.ValueOf(item), map[collectionKey]reflect.Value{}).Interface()
	}
	return cloned
}

type collectionKey struct {
	typeOf  reflect.Type
	pointer uintptr
}

func cloneCollection(value reflect.Value, seen map[collectionKey]reflect.Value) reflect.Value {
	if !value.IsValid() {
		return reflect.Zero(reflect.TypeFor[any]())
	}
	switch value.Kind() {
	case reflect.Interface:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		cloned := reflect.New(value.Type()).Elem()
		cloned.Set(cloneCollection(value.Elem(), seen))
		return cloned
	case reflect.Map:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		key := collectionKey{typeOf: value.Type(), pointer: value.Pointer()}
		if cloned, ok := seen[key]; ok {
			return cloned
		}
		cloned := reflect.MakeMapWithSize(value.Type(), value.Len())
		seen[key] = cloned
		iterator := value.MapRange()
		for iterator.Next() {
			cloned.SetMapIndex(iterator.Key(), cloneCollection(iterator.Value(), seen))
		}
		return cloned
	case reflect.Slice:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		key := collectionKey{typeOf: value.Type(), pointer: value.Pointer()}
		if cloned, ok := seen[key]; ok {
			return cloned
		}
		cloned := reflect.MakeSlice(value.Type(), value.Len(), value.Len())
		seen[key] = cloned
		for index := range value.Len() {
			cloned.Index(index).Set(cloneCollection(value.Index(index), seen))
		}
		return cloned
	case reflect.Array:
		cloned := reflect.New(value.Type()).Elem()
		for index := range value.Len() {
			cloned.Index(index).Set(cloneCollection(value.Index(index), seen))
		}
		return cloned
	default:
		return value
	}
}
