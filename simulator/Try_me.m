load_system('actuator.slx');
set_param('actuator',"SimulationCommand","disconnect");
%set_param('actuator',"SimulationCommand","connect");
set_param('actuator',"SimulationCommand","start");
%set_param('actuator',"SimulationCommand","continue");
set_param('actuator',"SimulationCommand","step");
set_param('actuator','SimulationCommand','pause');
%t = simlog_sscfluids_antagonistic_mckibben_actuator.Air_Muscle_Actuator_Top.p_I.series.values('MPa');
