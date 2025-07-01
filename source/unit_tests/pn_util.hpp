#include <state_observation/pn_model.hpp>
#include <state_observation/pointcloud_util.hpp>

namespace unittests
{
 /*
  * Factory for petri nets.
  * The structure of each net is visualized in the comment of each method.
  * 
  * P represents a place, T a transition, and M a token in parantheses along an arc
  * the numbers correspond to their indices in net.places, net.transitions respectively.
  *
  * Unless specified otherwise, all transitions forward the same token.
  */
	class pn_factory
	{
	public:
	
		/*
		 * P0 -> T0 -> P1 -> T2 -> P3
		 *  |                       ^
		 *  ---> T1 -> P2 -> T3 ----|
		 */
		static state_observation::pn_net::Ptr two_paths(const state_observation::object_parameters& object_params);

		/*
		 * P0 -> T0 -> P1 -> T1 -> P2 -|
		 *  ^                         T2
		 *  |- T4 <- P4 <- T3 <- P3 <--|
		 */
		static state_observation::pn_net::Ptr cycle(const state_observation::object_parameters& object_params);

		/*
		 *                                |-(M1)-> P4
		 *                                |
		 * P1 -(M1)> T1 -(M1)  |-(M1)---> T3 --|
		 *                v    |          |    | (side condition)
		 * P0 --> T0 --> P2 -----> T2 -> P3 <--|
		 */
		static state_observation::pn_net::Ptr two_block_stack(const state_observation::object_parameters& object_params);

		/*
		 * P0      ...       P(2^n) 
		 *                                
		 * ...               ...
		 *                         
		 * P(2^n-1)    ...     P(2^(n+1)-1)
		 */
		static state_observation::pn_net::Ptr omega_network(const state_observation::object_parameters& object_params,unsigned int n);
		
	    /*
		 * P0 -> T0 -> P2
		 * P1 -> T1 ---|
		 */
		//static state_observation::pn_net::Ptr interlink();

				/*
		 * Creates places + agents many net places where agent places are created first.
		 * Interconnects every place with every agent and vice versa for every token.
		 */
		static state_observation::pn_net::Ptr pick_and_place(const state_observation::object_parameters& object_params,
			unsigned int places, unsigned int tokens, unsigned int agents);
	};
}