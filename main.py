from environment import LoadBalancerEnv as env

if __name__ == "__main__":
    # Crear el entorno
    environment = env(n_max=10, max_steps=100, max_memory=1024)
    
    # Reiniciar el entorno para obtener el estado inicial
    state, info = environment.reset()
    print("Estado inicial:", state)
    print("Info:", info)
    
    # Ejemplo de acción (pesos de ruteo y decisión de auto-scaling)
    action = [0.1] * 10 + [0.8]  # Pesos iguales y decisión de escalar hacia arriba
    
    # Avanzar un paso en el entorno con la acción dada
    new_state, reward, terminated, truncated, info = environment.step(action)
    print("Nuevo estado:", new_state)
    print("Recompensa:", reward)
    print("¿Terminado?", terminated)
    print("¿Truncado?", truncated)
    print("Info:", info)